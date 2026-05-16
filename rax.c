/* Rax -- A radix tree implementation.
 *
 * Version 2.0 -- 18 March 2026
 *
 * Copyright (c) 2017-2026, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include "rax.h"

#ifndef RAX_MALLOC_INCLUDE
#define RAX_MALLOC_INCLUDE "rax_malloc.h"
#endif

#include RAX_MALLOC_INCLUDE

/* This is a special pointer that is guaranteed to never have the same value
 * of a radix tree node. It's used in order to report "not found" error without
 * requiring the function to have multiple return values. */
void *raxNotFound = (void*)"rax-not-found-pointer";

/* -------------------------------- Debugging ------------------------------ */

void raxDebugShowNode(const char *msg, raxNode *n);

/* Turn debugging messages on/off by compiling with RAX_DEBUG_MSG macro on.
 * When RAX_DEBUG_MSG is defined by default Rax operations will emit a lot
 * of debugging info to the standard output, however you can still turn
 * debugging on/off in order to enable it only when you suspect there is an
 * operation causing a bug using the function raxSetDebugMsg(). */
#ifdef RAX_DEBUG_MSG
#define debugf(...)                                                            \
    if (raxDebugMsg) {                                                         \
        printf("%s:%s:%d:\t", __FILE__, __func__, __LINE__);               \
        printf(__VA_ARGS__);                                                   \
        fflush(stdout);                                                        \
    }

#define debugnode(msg,n) raxDebugShowNode(msg,n)
#else
#define debugf(...)
#define debugnode(msg,n)
#endif

/* By default log debug info if RAX_DEBUG_MSG is defined. */
static int raxDebugMsg = 1;

/* When debug messages are enabled, turn them on/off dynamically. By
 * default they are enabled. Set the state to 0 to disable, and 1 to
 * re-enable. */
void raxSetDebugMsg(int onoff) {
    raxDebugMsg = onoff;
}

/* ------------------------- raxStack functions --------------------------
 * The raxStack is a simple stack of pointers that is capable of switching
 * from using a stack-allocated array to dynamic heap once a given number of
 * items are reached. It is used in order to retain the list of parent nodes
 * while walking the radix tree in order to implement certain operations that
 * need to navigate the tree upward.
 * ------------------------------------------------------------------------- */

/* Initialize the stack. */
static inline void raxStackInit(raxStack *ts) {
    ts->stack = ts->static_items;
    ts->items = 0;
    ts->maxitems = RAX_STACK_STATIC_ITEMS;
    ts->oom = 0;
}

/* Push an item into the stack, returns 1 on success, 0 on out of memory. */
static inline int raxStackPush(raxStack *ts, void *ptr) {
    if (ts->items == ts->maxitems) {
        if (ts->maxitems > SIZE_MAX / sizeof(void*) / 2) {
            ts->oom = 1;
            errno = ENOMEM;
            return 0;
        }
        if (ts->stack == ts->static_items) {
            ts->stack = rax_malloc(sizeof(void*)*ts->maxitems*2);
            if (ts->stack == NULL) {
                ts->stack = ts->static_items;
                ts->oom = 1;
                errno = ENOMEM;
                return 0;
            }
            memcpy(ts->stack,ts->static_items,sizeof(void*)*ts->maxitems);
        } else {
            void **newalloc = rax_realloc(ts->stack,sizeof(void*)*ts->maxitems*2);
            if (newalloc == NULL) {
                ts->oom = 1;
                errno = ENOMEM;
                return 0;
            }
            ts->stack = newalloc;
        }
        ts->maxitems *= 2;
    }
    ts->stack[ts->items] = ptr;
    ts->items++;
    return 1;
}

/* Pop an item from the stack, the function returns NULL if there are no
 * items to pop. */
static inline void *raxStackPop(raxStack *ts) {
    if (ts->items == 0) return NULL;
    ts->items--;
    return ts->stack[ts->items];
}

/* Return the stack item at the top of the stack without actually consuming
 * it. */
static inline void *raxStackPeek(raxStack *ts) {
    if (ts->items == 0) return NULL;
    return ts->stack[ts->items-1];
}

/* Free the stack in case we used heap allocation. */
static inline void raxStackFree(raxStack *ts) {
    if (ts->stack != ts->static_items) rax_free(ts->stack);
}

/* ----------------------------------------------------------------------------
 * Radix tree implementation
 * --------------------------------------------------------------------------*/

/* Return the padding needed in the characters section of a node having size
 * 'nodesize'. The padding is needed to store the child pointers to aligned
 * addresses. Note that we add 4 to the node size because the node has a four
 * bytes header. */
#define raxPadding(nodesize) ((sizeof(void*)-(((nodesize)+4) % sizeof(void*))) & (sizeof(void*)-1))

/* Return the pointer to the last child pointer in a node. For the compressed
 * nodes this is the only child pointer. */
#define raxNodeLastChildPtr(n) ((raxNode**) ( \
    ((char*)(n)) + \
    raxNodeCurrentLength(n) - \
    sizeof(raxNode*) - \
    (((n)->iskey && !(n)->isnull) ? sizeof(void*) : 0) \
))

/* Return the pointer to the first child pointer. */
#define raxNodeFirstChildPtr(n) ((raxNode**) ( \
    (n)->data + \
    (n)->size + \
    raxPadding((n)->size)))

/* Return the current total size of the node. Note that the second line
 * computes the padding after the string of characters, needed in order to
 * save pointers to aligned addresses. */
#define raxNodeCurrentLength(n) ( \
    sizeof(raxNode)+(n)->size+ \
    raxPadding((n)->size)+ \
    ((n)->iscompr ? sizeof(raxNode*) : sizeof(raxNode*)*(n)->size)+ \
    (((n)->iskey && !(n)->isnull)*sizeof(void*)) \
)

/* Return 1 if the child at index 'childidx' of node 'n' is an inline leaf
 * value rather than a pointer to a child node. Inline leaves are stored
 * directly in the child pointer slot to avoid allocating a separate leaf
 * node. For compressed nodes only child index 0 is meaningful. Children
 * at index >= 13 can never be inline (the bitmap only has 13 bits). */
static inline int raxIsInlineLeaf(raxNode *n, int childidx) {
    if (childidx < 0 || childidx >= 13) return 0;
    return (n->leafbitmap >> childidx) & 1;
}

raxNode *raxNewNode(size_t children, int datafield);
void raxSetData(raxNode *n, void *data);

/* Create a standalone leaf node storing the specified value. */
static inline raxNode *raxNewValueNode(void *data) {
    raxNode *leaf = raxNewNode(0,data != NULL);
    if (leaf == NULL) return NULL;
    leaf->iskey = 1;
    if (data != NULL) {
        leaf->isnull = 0;
        raxSetData(leaf,data);
    } else {
        leaf->isnull = 1;
    }
    return leaf;
}

/* Convert an inline leaf value stored in 'parentlink' into a real leaf node. */
static inline int raxMaterializeInlineLeaf(rax *rax, raxNode *parent,
                                           raxNode **parentlink,
                                           raxNode **childptr)
{
    void *value;
    memcpy(&value,parentlink,sizeof(value));
    raxNode *leaf = raxNewValueNode(value);
    if (leaf == NULL) return 0;
    memcpy(parentlink,&leaf,sizeof(leaf));
    int childidx = (int)(parentlink - raxNodeFirstChildPtr(parent));
    if (childidx < 13) parent->leafbitmap &= ~(1 << childidx);
    rax->numnodes++;
    if (childptr) *childptr = leaf;
    return 1;
}

/* Allocate a new non compressed node with the specified number of children.
 * If datafield is true, the allocation is made large enough to hold the
 * associated data pointer.
 * Returns the new node pointer. On out of memory NULL is returned. */
raxNode *raxNewNode(size_t children, int datafield) {
    size_t nodesize = sizeof(raxNode)+children+raxPadding(children)+
                      sizeof(raxNode*)*children;
    if (datafield) nodesize += sizeof(void*);
    raxNode *node = rax_malloc(nodesize);
    if (node == NULL) return NULL;
    node->iskey = 0;
    node->isnull = 0;
    node->iscompr = 0;
    node->leafbitmap = 0;
    node->size = children;
    return node;
}

/* Allocate a new rax and return its pointer. On out of memory the function
 * returns NULL. */
rax *raxNew(void) {
    rax *rax = rax_malloc(sizeof(*rax));
    if (rax == NULL) return NULL;
    rax->numele = 0;
    rax->numnodes = 1;
    rax->head = raxNewNode(0,0);
    if (rax->head == NULL) {
        rax_free(rax);
        return NULL;
    } else {
        return rax;
    }
}

/* realloc the node to make room for auxiliary data in order
 * to store an item in that node. On out of memory NULL is returned. */
raxNode *raxReallocForData(raxNode *n, void *data) {
    if (data == NULL) return n; /* No reallocation needed, setting isnull=1 */
    size_t curlen = raxNodeCurrentLength(n);
    return rax_realloc(n,curlen+sizeof(void*));
}

/* Set the node auxiliary data to the specified pointer. */
void raxSetData(raxNode *n, void *data) {
    n->iskey = 1;
    if (data != NULL) {
        n->isnull = 0;
        void **ndata = (void**)
            ((char*)n+raxNodeCurrentLength(n)-sizeof(void*));
        memcpy(ndata,&data,sizeof(data));
    } else {
        n->isnull = 1;
    }
}

/* Get the node auxiliary data. */
void *raxGetData(raxNode *n) {
    if (n->isnull) return NULL;
    void **ndata =(void**)((char*)n+raxNodeCurrentLength(n)-sizeof(void*));
    void *data;
    memcpy(&data,ndata,sizeof(data));
    return data;
}

/* Return the position where the edge 'c' should be inserted in order to
 * preserve lexicographic ordering. */
static inline int raxNodeFindChildPos(raxNode *n, unsigned char c) {
    int pos;

    assert(n->iscompr == 0);
    for (pos = 0; pos < n->size; pos++) {
        if (n->data[pos] > c) break;
    }
    return pos;
}

/* Like raxAddChild() but does not allocate the child node. Instead it
 * returns in 'parentlink' the address of the new child pointer, so that the
 * caller can store either a node pointer or an inline leaf value there. */
static inline raxNode *raxAddChildNoAlloc(rax *rax, raxNode *n,
                                          unsigned char c,
                                          raxNode ***parentlink)
{
    assert(n->iscompr == 0);

    size_t curlen = raxNodeCurrentLength(n);
    n->size++;
    size_t newlen = raxNodeCurrentLength(n);
    n->size--; /* For now restore the original size. We'll update it only on
                  success at the end. */

    /* Make space in the original node. */
    raxNode *newn = rax_realloc(n,newlen);
    if (newn == NULL) return NULL;
    n = newn;

    /* After the reallocation, we have up to 8/16 (depending on the system
     * pointer size, and the required node padding) bytes at the end, that is,
     * the additional char in the 'data' section, plus one pointer to the new
     * child, plus the padding needed in order to store addresses into aligned
     * locations.
     *
     * So if we start with the following node, having "abde" edges.
     *
     * Note:
     * - We assume 4 bytes pointer for simplicity.
     * - Each space below corresponds to one byte
     *
     * [HDR*][abde][Aptr][Bptr][Dptr][Eptr]|AUXP|
     *
     * After the reallocation we need: 1 byte for the new edge character
     * plus 4 bytes for a new child pointer (assuming 32 bit machine).
     * However after adding 1 byte to the edge char, the header + the edge
     * characters are no longer aligned, so we also need 3 bytes of padding.
     * In total the reallocation will add 1+4+3 bytes = 8 bytes:
     *
     * (Blank bytes are represented by ".")
     *
     * [HDR*][abde][Aptr][Bptr][Dptr][Eptr]|AUXP|[....][....]
     *
     * Let's find where to insert the new child in order to make sure
     * it is inserted in-place lexicographically. Assuming we are adding
     * a child "c" in our case pos will be = 2 after the end of the following
     * loop. */
    int pos = raxNodeFindChildPos(n,c);

    /* If child 12 is inline and we insert before or at it, that child
     * will move to slot 13. The bitmap cannot represent slot 13, so
     * pre-materialize it as a real node before we mutate the parent. */
    raxNode *materialized_child = NULL;
    if (pos <= 12 && raxIsInlineLeaf(n,12)) {
        void *value;
        raxNode **child12 = raxNodeFirstChildPtr(n)+12;
        memcpy(&value,child12,sizeof(value));
        materialized_child = raxNewValueNode(value);
        if (materialized_child == NULL) return NULL;
    }

    /* Now, if present, move auxiliary data pointer at the end
     * so that we can mess with the other data without overwriting it.
     * We will obtain something like that:
     *
     * [HDR*][abde][Aptr][Bptr][Dptr][Eptr][....][....]|AUXP|
     */
    unsigned char *src, *dst;
    if (n->iskey && !n->isnull) {
        src = ((unsigned char*)n+curlen-sizeof(void*));
        dst = ((unsigned char*)n+newlen-sizeof(void*));
        memmove(dst,src,sizeof(void*));
    }

    /* Compute the "shift", that is, how many bytes we need to move the
     * pointers section forward because of the addition of the new child
     * byte in the string section. Note that if we had no padding, that
     * would be always "1", since we are adding a single byte in the string
     * section of the node (where now there is "abde" basically).
     *
     * However we have padding, so it could be zero, or up to 8.
     *
     * Another way to think at the shift is, how many bytes we need to
     * move child pointers forward *other than* the obvious sizeof(void*)
     * needed for the additional pointer itself. */
    size_t shift = newlen - curlen - sizeof(void*);

    /* We said we are adding a node with edge 'c'. The insertion
     * point is between 'b' and 'd', so the 'pos' variable value is
     * the index of the first child pointer that we need to move forward
     * to make space for our new pointer.
     *
     * To start, move all the child pointers after the insertion point
     * of shift+sizeof(pointer) bytes on the right, to obtain:
     *
     * [HDR*][abde][Aptr][Bptr][....][....][Dptr][Eptr]|AUXP|
     */
    src = n->data+n->size+
          raxPadding(n->size)+
          sizeof(raxNode*)*pos;
    memmove(src+shift+sizeof(raxNode*),src,sizeof(raxNode*)*(n->size-pos));

    /* Move the pointers to the left of the insertion position as well. Often
     * we don't need to do anything if there was already some padding to use. In
     * that case the final destination of the pointers will be the same, however
     * in our example there was no pre-existing padding, so we added one byte
     * plus three bytes of padding. After the next memmove() things will look
     * like that:
     *
     * [HDR*][abde][....][Aptr][Bptr][....][Dptr][Eptr]|AUXP|
     */
    if (shift) {
        src = (unsigned char*) raxNodeFirstChildPtr(n);
        memmove(src+shift,src,sizeof(raxNode*)*pos);
    }

    /* Now make the space for the additional char in the data section,
     * but also move the pointers before the insertion point to the right
     * by shift bytes, in order to obtain the following:
     *
     * [HDR*][ab.d][e...][Aptr][Bptr][....][Dptr][Eptr]|AUXP|
     */
    src = n->data+pos;
    memmove(src+1,src,n->size-pos);

    /* We can now set the character and account for the additional child
     * pointer to get:
     *
     * [HDR*][abcd][e...][Aptr][Bptr][....][Dptr][Eptr]|AUXP|
     */
    n->data[pos] = c;
    n->size++;

    /* Shift leaf bitmap to account for the new child at position pos.
     * Bits at positions >= pos move up by one to make room. The new
     * child starts as a regular (non-inline) child. */
    if (n->leafbitmap && pos < 13) {
        uint16_t below = pos ? (n->leafbitmap & ((1u << pos) - 1)) : 0;
        uint16_t above = n->leafbitmap & ~below;
        n->leafbitmap = below | ((above << 1) & ((1u << 13) - 1));
    }
    src = (unsigned char*) raxNodeFirstChildPtr(n);
    if (materialized_child) {
        raxNode **spillfield = (raxNode**)(src+sizeof(raxNode*)*13);
        memcpy(spillfield,&materialized_child,sizeof(materialized_child));
        rax->numnodes++;
    }
    *parentlink = (raxNode**)(src+sizeof(raxNode*)*pos);
    return n;
}

/* Add a new child to the node 'n' representing the character 'c' and return
 * its new pointer, as well as the child pointer by reference. Additionally
 * '***parentlink' is populated with the raxNode pointer-to-pointer of where
 * the new child was stored, which is useful for the caller to replace the
 * child pointer if it gets reallocated.
 *
 * On success the new parent node pointer is returned (it may change because
 * of the realloc, so the caller should discard 'n' and use the new value).
 * On out of memory NULL is returned, and the old node is still valid. */
raxNode *raxAddChild(rax *rax, raxNode *n, unsigned char c, raxNode **childptr, raxNode ***parentlink) {
    /* Alloc the new child we will link to 'n'. */
    raxNode *child = raxNewNode(0,0);
    if (child == NULL) return NULL;

    raxNode *newn = raxAddChildNoAlloc(rax,n,c,parentlink);
    if (newn == NULL) {
        rax_free(child);
        return NULL;
    }
    memcpy(*parentlink,&child,sizeof(child));
    *childptr = child;
    return newn;
}

/* Turn the node 'n', that must be a node without any children, into a
 * compressed node representing a set of nodes linked one after the other
 * and having exactly one child each. The node can be a key or not: this
 * property and the associated value if any will be preserved.
 *
 * The function also returns a child node, since the last node of the
 * compressed chain cannot be part of the chain: it has zero children while
 * we can only compress inner nodes with exactly one child each. */
static inline raxNode *raxCompressNodeNoAlloc(raxNode *n, unsigned char *s,
                                              size_t len)
{
    assert(n->size == 0 && n->iscompr == 0);
    void *data = NULL; /* Initialized only to avoid warnings. */
    size_t newsize;

    debugf("Compress node: %.*s\n", (int)len,s);

    /* Make space in the parent node. */
    newsize = sizeof(raxNode)+len+raxPadding(len)+sizeof(raxNode*);
    if (n->iskey) {
        data = raxGetData(n); /* To restore it later. */
        if (!n->isnull) newsize += sizeof(void*);
    }
    raxNode *newn = rax_realloc(n,newsize);
    if (newn == NULL) return NULL;
    n = newn;

    n->iscompr = 1;
    n->size = len;
    memcpy(n->data,s,len);
    if (n->iskey) raxSetData(n,data);
    return n;
}

raxNode *raxCompressNode(raxNode *n, unsigned char *s, size_t len, raxNode **child) {
    /* Allocate the child to link to this node. */
    *child = raxNewNode(0,0);
    if (*child == NULL) return NULL;

    raxNode *newn = raxCompressNodeNoAlloc(n,s,len);
    if (newn == NULL) {
        rax_free(*child);
        return NULL;
    }
    n = newn;
    raxNode **childfield = raxNodeLastChildPtr(n);
    memcpy(childfield,child,sizeof(*child));
    return n;
}

/* Low level function that walks the tree looking for the string
 * 's' of 'len' bytes. The function returns the number of characters
 * of the key that was possible to process: if the returned integer
 * is the same as 'len', then it means that the node corresponding to the
 * string was found (however it may not be a key in case the node->iskey is
 * zero or if simply we stopped in the middle of a compressed node, so that
 * 'splitpos' is non zero).
 *
 * Otherwise if the returned integer is not the same as 'len', there was an
 * early stop during the tree walk because of a character mismatch.
 *
 * The node where the search ended (because the full string was processed
 * or because there was an early stop) is returned by reference as
 * '*stopnode' if the passed pointer is not NULL. This node link in the
 * parent's node is returned as '*plink' if not NULL. Finally, if the
 * search stopped in a compressed node, '*splitpos' returns the index
 * inside the compressed node where the search ended. This is useful to
 * know where to split the node for insertion.
 *
 * Note that when we stop in the middle of a compressed node with
 * a perfect match, this function will return a length equal to the
 * 'len' argument (all the key matched), and will return a *splitpos which is
 * always positive (that will represent the index of the character immediately
 * *after* the last match in the current compressed node).
 *
 * When instead we stop at a compressed node and *splitpos is zero, it
 * means that the current node represents the key (that is, none of the
 * compressed node characters are needed to represent the key, just all
 * its parents nodes). */
static inline size_t raxLowWalk(rax *rax, unsigned char *s, size_t len, raxNode **stopnode, raxNode ***plink, int *splitpos, raxStack *ts, int *inline_leaf) {
    raxNode *h = rax->head;
    raxNode **parentlink = &rax->head;

    if (inline_leaf) *inline_leaf = 0;
    size_t i = 0; /* Position in the string. */
    size_t j = 0; /* Position in the node children (or bytes if compressed).*/
    while(h->size && i < len) {
        debugnode("Lookup current node",h);
        unsigned char *v = h->data;

        if (h->iscompr) {
            for (j = 0; j < h->size && i < len; j++, i++) {
                if (v[j] != s[i]) break;
            }
            if (j != h->size) break;
        } else {
            /* Even when h->size is large, linear scan provides good
             * performances compared to other approaches that are in theory
             * more sounding, like performing a binary search. However
             * for nodes with many children, using memchr() is faster
             * since it is SIMD-accelerated on modern architectures. */
            if (h->size > 16) {
                unsigned char *found = memchr(v,s[i],h->size);
                if (found == NULL) break;
                j = found - v;
            } else {
                for (j = 0; j < h->size; j++) {
                    if (v[j] == s[i]) break;
                }
                if (j == h->size) break;
            }
            i++;
        }

        raxNode **children = raxNodeFirstChildPtr(h);
        if (h->iscompr) j = 0; /* Compressed node only child is at index 0. */

        /* If the child we are about to follow is an inline leaf (a value
         * stored directly in the child pointer slot), we can't descend
         * further. Stop the walk here: h remains as the parent, and
         * parentlink will point to the slot containing the inline value.
         * We do NOT push h onto the stack since we're not descending. */
        if (raxIsInlineLeaf(h,j)) {
            if (inline_leaf) *inline_leaf = 1;
            parentlink = children+j;
            break;
        }
        if (ts) raxStackPush(ts,h); /* Save stack of parent nodes. */
        memcpy(&h,children+j,sizeof(h));
        parentlink = children+j;
        j = 0; /* If the new node is compressed and we do not
                  iterate again (since i == l) set the split
                  position to 0 to signal this node represents
                  the searched key. */
    }
    debugnode("Lookup stop node is",h);
    if (stopnode) *stopnode = h;
    if (plink) *plink = parentlink;
    if (splitpos && h->iscompr) *splitpos = j;
    return i;
}

/* Insert the element 's' of size 'len', setting as auxiliary data
 * the pointer 'data'. If the element is already present, the associated
 * data is updated (only if 'overwrite' is set to 1), and 0 is returned,
 * otherwise the element is inserted and 1 is returned. On out of memory the
 * function returns 0 as well but sets errno to ENOMEM, otherwise errno will
 * be set to 0.
 */
int raxGenericInsert(rax *rax, unsigned char *s, size_t len, void *data, void **old, int overwrite) {
    size_t i;
    int j = 0; /* Split position. If raxLowWalk() stops in a compressed
                  node, the index 'j' represents the char we stopped within the
                  compressed node, that is, the position where to split the
                  node for insertion. */
    raxNode *h, **parentlink;

    debugf("### Insert %.*s with value %p\n", (int)len, s, data);
    int inline_leaf = 0;
    i = raxLowWalk(rax,s,len,&h,&parentlink,&j,NULL,&inline_leaf);

    /* If the key was found as an inline leaf, the value is stored
     * directly in the parent's child pointer slot. Update it in place
     * without any allocation. */
    if (i == len && inline_leaf) {
        void *curval;
        memcpy(&curval,parentlink,sizeof(curval));
        if (old) *old = curval;
        if (overwrite) memcpy(parentlink,&data,sizeof(data));
        errno = 0;
        return 0; /* Element already exists. */
    }

    /* If we stopped because we hit an inline leaf but still have
     * characters to insert, we must "un-inline" the leaf: allocate
     * a real node for it so we can continue the insertion. */
    if (inline_leaf && i < len) {
        raxNode *leaf;
        if (!raxMaterializeInlineLeaf(rax,h,parentlink,&leaf)) {
            errno = ENOMEM;
            return 0;
        }
        h = leaf;
        j = 0;
        /* h is now a real node with size=0 and iskey=1, iscompr=0.
         * Neither ALGO 1 nor ALGO 2 will trigger. We fall through to
         * the "insert remaining chars" loop. */
    }

    /* If i == len we walked following the whole string. If we are not
     * in the middle of a compressed node, the string is either already
     * inserted or this middle node is currently not a key, but can represent
     * our key. We have just to reallocate the node and make space for the
     * data pointer. */
    if (i == len && (!h->iscompr || j == 0 /* not in the middle if j is 0 */)) {
        debugf("### Insert: node representing key exists\n");
        /* Make space for the value pointer if needed. */
        if (!h->iskey || (h->isnull && overwrite)) {
            h = raxReallocForData(h,data);
            if (h) memcpy(parentlink,&h,sizeof(h));
        }
        if (h == NULL) {
            errno = ENOMEM;
            return 0;
        }

        /* Update the existing key if there is already one. */
        if (h->iskey) {
            if (old) *old = raxGetData(h);
            if (overwrite) raxSetData(h,data);
            errno = 0;
            return 0; /* Element already exists. */
        }

        /* Otherwise set the node as a key. Note that raxSetData()
         * will set h->iskey. */
        raxSetData(h,data);
        rax->numele++;
        return 1; /* Element inserted. */
    }

    /* If the node we stopped at is a compressed node, we need to
     * split it before to continue.
     *
     * Splitting a compressed node have a few possible cases.
     * Imagine that the node 'h' we are currently at is a compressed
     * node containing the string "ANNIBALE" (it means that it represents
     * nodes A -> N -> N -> I -> B -> A -> L -> E with the only child
     * pointer of this node pointing at the 'E' node, because remember that
     * we have characters at the edges of the graph, not inside the nodes
     * themselves.
     *
     * In order to show a real case imagine our node to also point to
     * another compressed node, that finally points at the node without
     * children, representing 'O':
     *
     *     "ANNIBALE" -> "SCO" -> []
     *
     * When inserting we may face the following cases. Note that all the cases
     * require the insertion of a non compressed node with exactly two
     * children, except for the last case which just requires splitting a
     * compressed node.
     *
     * 1) Inserting "ANNIENTARE"
     *
     *               |B| -> "ALE" -> "SCO" -> []
     *     "ANNI" -> |-|
     *               |E| -> (... continue algo ...) "NTARE" -> []
     *
     * 2) Inserting "ANNIBALI"
     *
     *                  |E| -> "SCO" -> []
     *     "ANNIBAL" -> |-|
     *                  |I| -> (... continue algo ...) []
     *
     * 3) Inserting "AGO" (Like case 1, but set iscompr = 0 into original node)
     *
     *            |N| -> "NIBALE" -> "SCO" -> []
     *     |A| -> |-|
     *            |G| -> (... continue algo ...) |O| -> []
     *
     * 4) Inserting "CIAO"
     *
     *     |A| -> "NNIBALE" -> "SCO" -> []
     *     |-|
     *     |C| -> (... continue algo ...) "IAO" -> []
     *
     * 5) Inserting "ANNI"
     *
     *     "ANNI" -> "BALE" -> "SCO" -> []
     *
     * The final algorithm for insertion covering all the above cases is as
     * follows.
     *
     * ============================= ALGO 1 =============================
     *
     * For the above cases 1 to 4, that is, all cases where we stopped in
     * the middle of a compressed node for a character mismatch, do:
     *
     * Let $SPLITPOS be the zero-based index at which, in the
     * compressed node array of characters, we found the mismatching
     * character. For example if the node contains "ANNIBALE" and we add
     * "ANNIENTARE" the $SPLITPOS is 4, that is, the index at which the
     * mismatching character is found.
     *
     * 1. Save the current compressed node $NEXT pointer (the pointer to the
     *    child element, that is always present in compressed nodes).
     *
     * 2. Create "split node" having as child the non common letter
     *    at the compressed node. The other non common letter (at the key)
     *    will be added later as we continue the normal insertion algorithm
     *    at step "6".
     *
     * 3a. IF $SPLITPOS == 0:
     *     Replace the old node with the split node, by copying the auxiliary
     *     data if any. Fix parent's reference. Free old node eventually
     *     (we still need its data for the next steps of the algorithm).
     *
     * 3b. IF $SPLITPOS != 0:
     *     Trim the compressed node (reallocating it as well) in order to
     *     contain $splitpos characters. Change child pointer in order to link
     *     to the split node. If new compressed node len is just 1, set
     *     iscompr to 0 (layout is the same). Fix parent's reference.
     *
     * 4a. IF the postfix len (the length of the remaining string of the
     *     original compressed node after the split character) is non zero,
     *     create a "postfix node". If the postfix node has just one character
     *     set iscompr to 0, otherwise iscompr to 1. Set the postfix node
     *     child pointer to $NEXT.
     *
     * 4b. IF the postfix len is zero, just use $NEXT as postfix pointer.
     *
     * 5. Set child[0] of split node to postfix node.
     *
     * 6. Set the split node as the current node, set current index at child[1]
     *    and continue insertion algorithm as usually.
     *
     * ============================= ALGO 2 =============================
     *
     * For case 5, that is, if we stopped in the middle of a compressed
     * node but no mismatch was found, do:
     *
     * Let $SPLITPOS be the zero-based index at which, in the
     * compressed node array of characters, we stopped iterating because
     * there were no more keys character to match. So in the example of
     * the node "ANNIBALE", adding the string "ANNI", the $SPLITPOS is 4.
     *
     * 1. Save the current compressed node $NEXT pointer (the pointer to the
     *    child element, that is always present in compressed nodes).
     *
     * 2. Create a "postfix node" containing all the characters from $SPLITPOS
     *    to the end. Use $NEXT as the postfix node child pointer.
     *    If the postfix node length is 1, set iscompr to 0.
     *    Set the node as a key with the associated value of the new
     *    inserted key.
     *
     * 3. Trim the current node to contain the first $SPLITPOS characters.
     *    As usually if the new node length is just 1, set iscompr to 0.
     *    Take the iskey / associated value as it was in the original node.
     *    Fix the parent's reference.
     *
     * 4. Set the postfix node as the only child pointer of the trimmed
     *    node created at step 1.
     */

    /* ------------------------- ALGORITHM 1 --------------------------- */
    if (h->iscompr && i != len) {
        debugf("ALGO 1: Stopped at compressed node %.*s (%p)\n",
            h->size, h->data, (void*)h);
        debugf("Still to insert: %.*s\n", (int)(len-i), s+i);
        debugf("Splitting at %d: '%c'\n", j, ((char*)h->data)[j]);
        debugf("Other (key) letter is '%c'\n", s[i]);

        /* 1: Save next pointer (or inline value if the child was inlined). */
        raxNode **childfield = raxNodeLastChildPtr(h);
        raxNode *next;
        memcpy(&next,childfield,sizeof(next));
        int next_is_inline = h->leafbitmap & 1;
        debugf("Next is %p (inline=%d)\n", (void*)next, next_is_inline);
        debugf("iskey %d\n", h->iskey);
        if (h->iskey) {
            debugf("key value is %p\n", raxGetData(h));
        }

        /* Set the length of the additional nodes we will need. */
        size_t trimmedlen = j;
        size_t postfixlen = h->size - j - 1;
        int split_node_is_key = !trimmedlen && h->iskey && !h->isnull;
        size_t nodesize;

        /* 2: Create the split node. Also allocate the other nodes we'll need
         *    ASAP, so that it will be simpler to handle OOM. */
        raxNode *splitnode = raxNewNode(1, split_node_is_key);
        raxNode *trimmed = NULL;
        raxNode *postfix = NULL;

        if (trimmedlen) {
            nodesize = sizeof(raxNode)+trimmedlen+raxPadding(trimmedlen)+
                       sizeof(raxNode*);
            if (h->iskey && !h->isnull) nodesize += sizeof(void*);
            trimmed = rax_malloc(nodesize);
        }

        if (postfixlen) {
            nodesize = sizeof(raxNode)+postfixlen+raxPadding(postfixlen)+
                       sizeof(raxNode*);
            postfix = rax_malloc(nodesize);
        }

        /* OOM? Abort now that the tree is untouched. */
        if (splitnode == NULL ||
            (trimmedlen && trimmed == NULL) ||
            (postfixlen && postfix == NULL))
        {
            rax_free(splitnode);
            rax_free(trimmed);
            rax_free(postfix);
            errno = ENOMEM;
            return 0;
        }
        splitnode->data[0] = h->data[j];

        if (j == 0) {
            /* 3a: Replace the old node with the split node. */
            if (h->iskey) {
                void *ndata = raxGetData(h);
                raxSetData(splitnode,ndata);
            }
            memcpy(parentlink,&splitnode,sizeof(splitnode));
        } else {
            /* 3b: Trim the compressed node. */
            trimmed->size = j;
            memcpy(trimmed->data,h->data,j);
            trimmed->iscompr = j > 1 ? 1 : 0;
            trimmed->iskey = h->iskey;
            trimmed->isnull = h->isnull;
            trimmed->leafbitmap = 0;
            if (h->iskey && !h->isnull) {
                void *ndata = raxGetData(h);
                raxSetData(trimmed,ndata);
            }
            raxNode **cp = raxNodeLastChildPtr(trimmed);
            memcpy(cp,&splitnode,sizeof(splitnode));
            memcpy(parentlink,&trimmed,sizeof(trimmed));
            parentlink = cp; /* Set parentlink to splitnode parent. */
            rax->numnodes++;
        }

        /* 4: Create the postfix node: what remains of the original
         * compressed node after the split. */
        if (postfixlen) {
            /* 4a: create a postfix node. */
            postfix->iskey = 0;
            postfix->isnull = 0;
            postfix->size = postfixlen;
            postfix->iscompr = postfixlen > 1;
            postfix->leafbitmap = next_is_inline ? 1 : 0;
            memcpy(postfix->data,h->data+j+1,postfixlen);
            raxNode **cp = raxNodeLastChildPtr(postfix);
            memcpy(cp,&next,sizeof(next));
            rax->numnodes++;
        } else {
            /* 4b: just use next as postfix node. */
            postfix = next;
        }

        /* 5: Set splitnode first child as the postfix node.
         *    If postfixlen was 0, postfix is actually 'next' which may be
         *    an inline value. In that case propagate the inline status. */
        raxNode **splitchild = raxNodeLastChildPtr(splitnode);
        memcpy(splitchild,&postfix,sizeof(postfix));
        if (!postfixlen && next_is_inline)
            splitnode->leafbitmap = 1;

        /* 6. Continue insertion: this will cause the splitnode to
         * get a new child (the non common character at the currently
         * inserted key). */
        rax_free(h);
        h = splitnode;
    } else if (h->iscompr && i == len) {
    /* ------------------------- ALGORITHM 2 --------------------------- */
        debugf("ALGO 2: Stopped at compressed node %.*s (%p) j = %d\n",
            h->size, h->data, (void*)h, j);

        /* Allocate postfix & trimmed nodes ASAP to fail for OOM gracefully. */
        size_t postfixlen = h->size - j;
        size_t nodesize = sizeof(raxNode)+postfixlen+raxPadding(postfixlen)+
                          sizeof(raxNode*);
        if (data != NULL) nodesize += sizeof(void*);
        raxNode *postfix = rax_malloc(nodesize);

        nodesize = sizeof(raxNode)+j+raxPadding(j)+sizeof(raxNode*);
        if (h->iskey && !h->isnull) nodesize += sizeof(void*);
        raxNode *trimmed = rax_malloc(nodesize);

        if (postfix == NULL || trimmed == NULL) {
            rax_free(postfix);
            rax_free(trimmed);
            errno = ENOMEM;
            return 0;
        }

        /* 1: Save next pointer (or inline value). */
        raxNode **childfield = raxNodeLastChildPtr(h);
        raxNode *next;
        memcpy(&next,childfield,sizeof(next));
        int next_is_inline = h->leafbitmap & 1;

        /* 2: Create the postfix node. */
        postfix->size = postfixlen;
        postfix->iscompr = postfixlen > 1;
        postfix->iskey = 1;
        postfix->isnull = 0;
        postfix->leafbitmap = next_is_inline ? 1 : 0;
        memcpy(postfix->data,h->data+j,postfixlen);
        raxSetData(postfix,data);
        raxNode **cp = raxNodeLastChildPtr(postfix);
        memcpy(cp,&next,sizeof(next));
        rax->numnodes++;

        /* 3: Trim the compressed node. */
        trimmed->size = j;
        trimmed->iscompr = j > 1;
        trimmed->iskey = 0;
        trimmed->isnull = 0;
        trimmed->leafbitmap = 0;
        memcpy(trimmed->data,h->data,j);
        memcpy(parentlink,&trimmed,sizeof(trimmed));
        if (h->iskey) {
            void *aux = raxGetData(h);
            raxSetData(trimmed,aux);
        }

        /* Fix the trimmed node child pointer to point to
         * the postfix node. */
        cp = raxNodeLastChildPtr(trimmed);
        memcpy(cp,&postfix,sizeof(postfix));

        /* Finish! We don't need to continue with the insertion
         * algorithm for ALGO 2. The key is already inserted. */
        rax->numele++;
        rax_free(h);
        return 1; /* Key inserted. */
    }

    /* We walked the radix tree as far as we could, but still there are left
     * chars in our string. We need to insert the missing nodes. */
    while(i < len) {
        raxNode *child;

        /* If this node is going to have a single child, and there
         * are other characters, so that that would result in a chain
         * of single-childed nodes, turn it into a compressed node. */
        if (h->size == 0 && len-i > 1) {
            debugf("Inserting compressed node\n");
            size_t comprsize = len-i;
            if (comprsize > RAX_NODE_MAX_SIZE)
                comprsize = RAX_NODE_MAX_SIZE;
            if (comprsize == len-i) {
                raxNode *newh = raxCompressNodeNoAlloc(h,s+i,comprsize);
                if (newh == NULL) goto oom;
                h = newh;
                memcpy(parentlink,&h,sizeof(h));
                parentlink = raxNodeLastChildPtr(h);
                memcpy(parentlink,&data,sizeof(data));
                h->leafbitmap = 1;
                rax->numele++;
                return 1; /* Element inserted. */
            }
            raxNode *newh = raxCompressNode(h,s+i,comprsize,&child);
            if (newh == NULL) goto oom;
            h = newh;
            memcpy(parentlink,&h,sizeof(h));
            parentlink = raxNodeLastChildPtr(h);
            i += comprsize;
        } else {
            debugf("Inserting normal node\n");
            if (len-i == 1 && raxNodeFindChildPos(h,s[i]) < 13) {
                raxNode **new_parentlink;
                raxNode *newh = raxAddChildNoAlloc(rax,h,s[i],&new_parentlink);
                if (newh == NULL) goto oom;
                h = newh;
                memcpy(parentlink,&h,sizeof(h));
                parentlink = new_parentlink;
                int childidx = (int)(parentlink - raxNodeFirstChildPtr(h));
                memcpy(parentlink,&data,sizeof(data));
                h->leafbitmap |= (1 << childidx);
                rax->numele++;
                return 1; /* Element inserted. */
            }
            raxNode **new_parentlink;
            raxNode *newh = raxAddChild(rax,h,s[i],&child,&new_parentlink);
            if (newh == NULL) goto oom;
            h = newh;
            memcpy(parentlink,&h,sizeof(h));
            parentlink = new_parentlink;
            i++;
        }
        rax->numnodes++;
        h = child;
    }
    raxNode *newh = raxReallocForData(h,data);
    if (newh == NULL) goto oom;
    h = newh;
    if (!h->iskey) rax->numele++;
    raxSetData(h,data);
    memcpy(parentlink,&h,sizeof(h));
    return 1; /* Element inserted. */

oom:
    /* This code path handles out of memory after part of the sub-tree was
     * already modified. Set the node as a key, and then remove it. However we
     * do that only if the node is a terminal node, otherwise if the OOM
     * happened reallocating a node in the middle, we don't need to free
     * anything. */
    if (h->size == 0) {
        h->isnull = 1;
        h->iskey = 1;
        rax->numele++; /* Compensate the next remove. */
        assert(raxRemove(rax,s,i,NULL) != 0);
    }
    errno = ENOMEM;
    return 0;
}

/* Overwriting insert. Just a wrapper for raxGenericInsert() that will
 * update the element if there is already one for the same key. */
int raxInsert(rax *rax, unsigned char *s, size_t len, void *data, void **old) {
    return raxGenericInsert(rax,s,len,data,old,1);
}

/* Non overwriting insert function: this if an element with the same key
 * exists, the value is not updated and the function returns 0.
 * This is a just a wrapper for raxGenericInsert(). */
int raxTryInsert(rax *rax, unsigned char *s, size_t len, void *data, void **old) {
    return raxGenericInsert(rax,s,len,data,old,0);
}

/* Find a key in the rax, returns raxNotFound special void pointer value
 * if the item was not found, otherwise the value associated with the
 * item is returned. */
void *raxFind(rax *rax, unsigned char *s, size_t len) {
    raxNode *h;

    debugf("### Lookup: %.*s\n", (int)len, s);
    int splitpos = 0;
    int inline_leaf = 0;
    raxNode **parentlink;
    size_t i = raxLowWalk(rax,s,len,&h,&parentlink,&splitpos,NULL,&inline_leaf);
    if (inline_leaf && i == len) {
        void *val;
        memcpy(&val,parentlink,sizeof(val));
        return val;
    }
    if (i != len || (h->iscompr && splitpos != 0) || !h->iskey)
        return raxNotFound;
    return raxGetData(h);
}

/* Return the memory address where the 'parent' node stores the specified
 * 'child' pointer, so that the caller can update the pointer with another
 * one if needed. The function assumes it will find a match, otherwise the
 * operation is an undefined behavior (it will continue scanning the
 * memory without any bound checking). */
raxNode **raxFindParentLink(raxNode *parent, raxNode *child) {
    raxNode **cp = raxNodeFirstChildPtr(parent);
    raxNode *c;
    while(1) {
        memcpy(&c,cp,sizeof(c));
        if (c == child) break;
        cp++;
    }
    return cp;
}

/* Low level child removal from node. 'childptr' must point to the child
 * pointer stored inside the parent node, and is used directly instead of
 * searching by child value. The new node pointer (after the child
 * removal) is returned. Note that this function does not fix the pointer
 * of the parent node in its parent, so this task is up to the caller.
 * The function never fails for out of memory. */
static inline raxNode *raxRemoveChildAtPtr(raxNode *parent, raxNode **childptr) {
    debugnode("raxRemoveChild before", parent);
    /* If parent is a compressed node (having a single child, as for definition
     * of the data structure), the removal of the child consists into turning
     * it into a normal node without children. */
    if (parent->iscompr) {
        void *data = NULL;
        if (parent->iskey) data = raxGetData(parent);
        parent->isnull = 0;
        parent->iscompr = 0;
        parent->size = 0;
        parent->leafbitmap = 0;
        if (parent->iskey) raxSetData(parent,data);
        debugnode("raxRemoveChild after", parent);
        return parent;
    }

    /* Otherwise we need to scan for the child pointer and memmove()
     * accordingly.
     *
     * 1. To start we seek the first element in both the children
     *    pointers and edge bytes in the node. */
    raxNode **cp = raxNodeFirstChildPtr(parent);
    raxNode **c = childptr;
    unsigned char *e = parent->data + (c - cp);

    /* 3. Remove the edge and the pointer by memmoving the remaining children
     *    pointer and edge bytes one position before. */
    int taillen = parent->size - (e - parent->data) - 1;
    debugf("raxRemoveChild tail len: %d\n", taillen);
    memmove(e,e+1,taillen);

    /* Compute the shift, that is the amount of bytes we should move our
     * child pointers to the left, since the removal of one edge character
     * and the corresponding padding change, may change the layout.
     * We just check if in the old version of the node there was at the
     * end just a single byte and all padding: in that case removing one char
     * will remove a whole sizeof(void*) word. */
    size_t shift = ((parent->size+4) % sizeof(void*)) == 1 ? sizeof(void*) : 0;

    /* Move the children pointers before the deletion point. */
    if (shift)
        memmove(((char*)cp)-shift,cp,(parent->size-taillen-1)*sizeof(raxNode**));

    /* Move the remaining "tail" pointers at the right position as well. */
    size_t valuelen = (parent->iskey && !parent->isnull) ? sizeof(void*) : 0;
    memmove(((char*)c)-shift,c+1,taillen*sizeof(raxNode**)+valuelen);

    /* 4. Update size and shift the leaf bitmap accordingly. Bits above
     *    the removed position shift down by one. */
    int pos = (int)(e - parent->data);
    parent->size--;
    if (parent->leafbitmap && pos < 13) {
        uint16_t above = parent->leafbitmap & ~((1u << (pos+1)) - 1);
        uint16_t below = pos ? (parent->leafbitmap & ((1u << pos) - 1)) : 0;
        parent->leafbitmap = below | (above >> 1);
    }

    /* We don't realloc the node to its new size: the node is already
     * consistent with the updated size, and shrinking reallocs rarely
     * release memory due to allocator bucketing. Skipping the realloc
     * avoids the overhead of a system call that almost never helps. */
    debugnode("raxRemoveChild after", parent);
    return parent;
}

/* Low level child removal from node. The new node pointer (after the child
 * removal) is returned. Note that this function does not fix the pointer
 * of the parent node in its parent, so this task is up to the caller.
 * The function never fails for out of memory. */
raxNode *raxRemoveChild(raxNode *parent, raxNode *child) {
    raxNode **cp = raxNodeFirstChildPtr(parent);
    while(1) {
        raxNode *aux;
        memcpy(&aux,cp,sizeof(aux));
        if (aux == child) break;
        cp++;
    }
    return raxRemoveChildAtPtr(parent,cp);
}

/* Free the useless node 'h' that was left after a deletion, and keep moving
 * upward while the parent would also become a non-key single-child node.
 * The returned node is the first one that remains in the tree and may
 * require recompression. */
static inline raxNode *raxRemoveCleanup(rax *rax, raxNode *h, raxStack *ts,
                                        int *trycompress)
{
    raxNode *child = NULL;

    while(h != rax->head) {
        child = h;
        debugf("Freeing child %p [%.*s] key:%d\n", (void*)child,
            (int)child->size, (char*)child->data, child->iskey);
        rax_free(child);
        rax->numnodes--;
        h = raxStackPop(ts);
         /* If this node has more then one child, or actually holds
          * a key, stop here. */
        if (h->iskey || (!h->iscompr && h->size != 1)) break;
    }
    if (child) {
        debugf("Unlinking child %p from parent %p\n",
            (void*)child, (void*)h);
        raxNode *new = raxRemoveChild(h,child);
        if (new != h) {
            raxNode *parent = raxStackPeek(ts);
            raxNode **parentlink;
            if (parent == NULL) {
                parentlink = &rax->head;
            } else {
                parentlink = raxFindParentLink(parent,h);
            }
            memcpy(parentlink,&new,sizeof(new));
        }

        /* If after the removal the node has just a single child
         * and is not a key, we need to try to compress it. */
        if (new->size == 1 && new->iskey == 0) {
            *trycompress = 1;
            h = new;
        }
    }
    return h;
}

/* Remove the specified item. Returns 1 if the item was found and
 * deleted, 0 otherwise. */
int raxRemove(rax *rax, unsigned char *s, size_t len, void **old) {
    raxNode *h;
    raxNode **parentlink;
    raxStack ts;

    debugf("### Delete: %.*s\n", (int)len, s);
    raxStackInit(&ts);
    int splitpos = 0;
    int inline_leaf = 0;
    size_t i = raxLowWalk(rax,s,len,&h,&parentlink,&splitpos,&ts,&inline_leaf);
    int trycompress = 0; /* Will be set to 1 if we should try to optimize the
                            tree resulting from the deletion. */

    /* Inline leaves can be deleted directly, without materializing a
     * temporary node. */
    if (i == len && inline_leaf) {
        void *val;
        memcpy(&val,parentlink,sizeof(val));
        if (old) *old = val;
        rax->numele--;
        h = raxRemoveChildAtPtr(h,parentlink);
        if (h->size == 0 && h->iskey == 0) {
            debugf("Key deleted as inline leaf. Cleanup needed.\n");
            h = raxRemoveCleanup(rax,h,&ts,&trycompress);
        } else if (h->size == 1 && h->iskey == 0) {
            trycompress = 1;
        }
        goto postdelete;
    }

    if (i != len || (h->iscompr && splitpos != 0) || !h->iskey) {
        raxStackFree(&ts);
        return 0;
    }
    if (old) *old = raxGetData(h);
    h->iskey = 0;
    rax->numele--;

    /* If this node has no children, the deletion needs to reclaim the
     * no longer used nodes. This is an iterative process that needs to
     * walk the three upward, deleting all the nodes with just one child
     * that are not keys, until the head of the rax is reached or the first
     * node with more than one child is found. */

    if (h->size == 0) {
        debugf("Key deleted in node without children. Cleanup needed.\n");
        h = raxRemoveCleanup(rax,h,&ts,&trycompress);
    } else if (h->size == 1) {
        /* If the node had just one child, after the removal of the key
         * further compression with adjacent nodes is potentially possible. */
        trycompress = 1;
    }

postdelete:
    /* Don't try node compression if our nodes pointers stack is not
     * complete because of OOM while executing raxLowWalk() */
    if (trycompress && ts.oom) trycompress = 0;

    /* Recompression: if trycompress is true, 'h' points to a radix tree node
     * that changed in a way that could allow to compress nodes in this
     * sub-branch. Compressed nodes represent chains of nodes that are not
     * keys and have a single child, so there are two deletion events that
     * may alter the tree so that further compression is needed:
     *
     * 1) A node with a single child was a key and now no longer is a key.
     * 2) A node with two children now has just one child.
     *
     * We try to navigate upward till there are other nodes that can be
     * compressed, when we reach the upper node which is not a key and has
     * a single child, we scan the chain of children to collect the
     * compressable part of the tree, and replace the current node with the
     * new one, fixing the child pointer to reference the first non
     * compressable node.
     *
     * Example of case "1". A tree stores the keys "FOO" = 1 and
     * "FOOBAR" = 2:
     *
     *
     * "FOO" -> "BAR" -> [] (2)
     *           (1)
     *
     * After the removal of "FOO" the tree can be compressed as:
     *
     * "FOOBAR" -> [] (2)
     *
     *
     * Example of case "2". A tree stores the keys "FOOBAR" = 1 and
     * "FOOTER" = 2:
     *
     *          |B| -> "AR" -> [] (1)
     * "FOO" -> |-|
     *          |T| -> "ER" -> [] (2)
     *
     * After the removal of "FOOTER" the resulting tree is:
     *
     * "FOO" -> |B| -> "AR" -> [] (1)
     *
     * That can be compressed into:
     *
     * "FOOBAR" -> [] (1)
     */
    if (trycompress) {
        debugf("After removing %.*s:\n", (int)len, s);
        debugnode("Compression may be needed",h);
        debugf("Seek start node\n");

        /* Try to reach the upper node that is compressible.
         * At the end of the loop 'h' will point to the first node we
         * can try to compress and 'parent' to its parent. */
        raxNode *parent;
        while(1) {
            parent = raxStackPop(&ts);
            if (!parent || parent->iskey ||
                (!parent->iscompr && parent->size != 1)) break;
            h = parent;
            debugnode("Going up to",h);
        }
        raxNode *start = h; /* Compression starting node. */

        /* Scan chain of nodes we can compress. */
        size_t comprsize = h->size;
        int nodes = 1;
        while(h->size != 0) {
            raxNode **cp = raxNodeLastChildPtr(h);
            int lastidx = h->iscompr ? 0 : h->size-1;
            if (raxIsInlineLeaf(h,lastidx)) break; /* Can't follow inline. */
            memcpy(&h,cp,sizeof(h));
            if (h->iskey || (!h->iscompr && h->size != 1)) break;
            /* Stop here if going to the next node would result into
             * a compressed node larger than h->size can hold. */
            if (comprsize + h->size > RAX_NODE_MAX_SIZE) break;
            nodes++;
            comprsize += h->size;
        }
        if (nodes > 1) {
            /* If we can compress, create the new node and populate it. */
            size_t nodesize =
                sizeof(raxNode)+comprsize+raxPadding(comprsize)+sizeof(raxNode*);
            raxNode *new = rax_malloc(nodesize);
            /* An out of memory here just means we cannot optimize this
             * node, but the tree is left in a consistent state. */
            if (new == NULL) {
                raxStackFree(&ts);
                return 1;
            }
            new->iskey = 0;
            new->isnull = 0;
            new->iscompr = 1;
            new->leafbitmap = 0;
            new->size = comprsize;
            rax->numnodes++;

            /* Scan again, this time to populate the new node content and
             * to fix the new node child pointer. At the same time we free
             * all the nodes that we'll no longer use. */
            comprsize = 0;
            h = start;
            while(h->size != 0) {
                memcpy(new->data+comprsize,h->data,h->size);
                comprsize += h->size;
                raxNode **cp = raxNodeLastChildPtr(h);
                int lastidx = h->iscompr ? 0 : h->size-1;
                raxNode *tofree = h;
                if (raxIsInlineLeaf(h,lastidx)) {
                    /* Read the inline value before freeing the node,
                     * since cp points into it. */
                    memcpy(&h,cp,sizeof(h));
                    rax_free(tofree); rax->numnodes--;
                    new->leafbitmap = 1;
                    break;
                }
                memcpy(&h,cp,sizeof(h));
                rax_free(tofree); rax->numnodes--;
                if (h->iskey || (!h->iscompr && h->size != 1)) break;
            }
            debugnode("New node",new);

            /* Now 'h' points to the first node that we still need to use,
             * so our new node child pointer will point to it. */
            raxNode **cp = raxNodeLastChildPtr(new);
            memcpy(cp,&h,sizeof(h));

            /* Fix parent link. */
            if (parent) {
                raxNode **parentlink = raxFindParentLink(parent,start);
                memcpy(parentlink,&new,sizeof(new));
            } else {
                rax->head = new;
            }

            debugf("Compressed %d nodes, %d total bytes\n",
                nodes, (int)comprsize);
        }
    }
    raxStackFree(&ts);
    return 1;
}

/* This is the core of raxFree(): performs a depth-first scan of the
 * tree and releases all the nodes found. */
void raxRecursiveFree(rax *rax, raxNode *n, void (*free_callback)(void*)) {
    debugnode("free traversing",n);
    int numchildren = n->iscompr ? 1 : n->size;
    raxNode **cp = raxNodeLastChildPtr(n);
    while(numchildren--) {
        if (raxIsInlineLeaf(n,numchildren)) {
            /* Inline leaf: the slot contains a value, not a node pointer.
             * Call the free callback on the value but don't recurse. */
            if (free_callback) {
                void *val;
                memcpy(&val,cp,sizeof(val));
                if (val != NULL) free_callback(val);
            }
        } else {
            raxNode *child;
            memcpy(&child,cp,sizeof(child));
            raxRecursiveFree(rax,child,free_callback);
        }
        cp--;
    }
    debugnode("free depth-first",n);
    if (free_callback && n->iskey && !n->isnull)
        free_callback(raxGetData(n));
    rax_free(n);
    rax->numnodes--;
}

/* Free a whole radix tree, calling the specified callback in order to
 * free the auxiliary data. */
void raxFreeWithCallback(rax *rax, void (*free_callback)(void*)) {
    raxRecursiveFree(rax,rax->head,free_callback);
    assert(rax->numnodes == 0);
    rax_free(rax);
}

/* Free a whole radix tree. */
void raxFree(rax *rax) {
    raxFreeWithCallback(rax,NULL);
}

/* ------------------------------- Iterator --------------------------------- */

/* Initialize a Rax iterator. This call should be performed a single time
 * to initialize the iterator, and must be followed by a raxSeek() call,
 * otherwise the raxPrev()/raxNext() functions will just return EOF. */
void raxStart(raxIterator *it, rax *rt) {
    it->flags = RAX_ITER_EOF; /* No crash if the iterator is not seeked. */
    it->rt = rt;
    it->key_len = 0;
    it->key = it->key_static_string;
    it->key_max = RAX_ITER_STATIC_LEN;
    it->data = NULL;
    it->node_child = -1;
    it->node_cb = NULL;
    raxStackInit(&it->stack);
}

/* Append characters at the current key string of the iterator 'it'. This
 * is a low level function used to implement the iterator, not callable by
 * the user. Returns 0 on out of memory, otherwise 1 is returned. */
int raxIteratorAddChars(raxIterator *it, unsigned char *s, size_t len) {
    if (len == 0) return 1;
    if (it->key_max < it->key_len+len) {
        int from_static = it->key == it->key_static_string;
        unsigned char *old = from_static ? NULL : it->key;
        size_t new_max = (it->key_len+len)*2;
        unsigned char *new_key = rax_realloc(old,new_max);
        if (new_key == NULL) {
            errno = ENOMEM;
            return 0;
        }
        it->key = new_key;
        if (from_static) memcpy(it->key,it->key_static_string,it->key_len);
        it->key_max = new_max;
    }
    /* Use memmove since there could be an overlap between 's' and
     * it->key when we use the current key in order to re-seek. */
    memmove(it->key+it->key_len,s,len);
    it->key_len += len;
    return 1;
}

/* Remove the specified number of chars from the right of the current
 * iterator key. */
void raxIteratorDelChars(raxIterator *it, size_t count) {
    it->key_len -= count;
}

static inline int raxIteratorIsInlineLeaf(raxIterator *it) {
    return it->flags & RAX_ITER_INLINE_LEAF;
}

static inline void raxIteratorClearInlineLeaf(raxIterator *it) {
    it->flags &= ~RAX_ITER_INLINE_LEAF;
}

static inline int raxIteratorSetInlineLeaf(raxIterator *it, raxNode *parent,
                                           raxNode **childfield, int childidx)
{
    memcpy(&it->data,childfield,sizeof(it->data));
    it->node = parent;
    it->node_child = childidx;
    it->flags |= RAX_ITER_INLINE_LEAF;
    return 1;
}

/* Descend from 'parent' into child 'childidx', updating the iterator key.
 * If the child is inline we stop on the virtual leaf without changing the
 * tree. Otherwise we enter the real child node as usual. */
static inline int raxIteratorEnterChild(raxIterator *it, raxNode *parent,
                                        raxNode **childfield, int childidx)
{
    if (parent->iscompr) {
        if (!raxIteratorAddChars(it,parent->data,parent->size)) return 0;
    } else {
        if (!raxIteratorAddChars(it,parent->data+childidx,1)) return 0;
    }

    if (raxIsInlineLeaf(parent,childidx))
        return raxIteratorSetInlineLeaf(it,parent,childfield,childidx);

    if (!raxStackPush(&it->stack,parent)) return 0;
    memcpy(&it->node,childfield,sizeof(it->node));
    it->node_child = childidx;
    if (it->node_cb && it->node_cb(&it->node))
        memcpy(childfield,&it->node,sizeof(it->node));
    it->data = it->node->iskey ? raxGetData(it->node) : NULL;
    return 1;
}

/* Return the pointer-to-pointer in the tree that references the element
 * currently selected by the iterator. For regular key nodes this is the
 * parent link (or the tree head itself). For inline leaves it is the child
 * slot inside the parent node holding the raw value pointer. */
static raxNode **raxIteratorCurrentParentLink(raxIterator *it, raxNode **parent) {
    raxNode *p;
    raxNode **cp;
    int childidx, numchildren;

    if (raxIteratorIsInlineLeaf(it)) {
        p = it->node;
        if (p == NULL) return NULL;
        numchildren = p->iscompr ? 1 : p->size;
        childidx = it->node_child;
        if (childidx < 0 || childidx >= numchildren ||
            !raxIsInlineLeaf(p,childidx))
        {
            if (p->iscompr) {
                childidx = 0;
            } else {
                if (it->key_len == 0) return NULL;
                unsigned char c = it->key[it->key_len-1];
                for (childidx = 0; childidx < numchildren; childidx++) {
                    if (p->data[childidx] == c && raxIsInlineLeaf(p,childidx))
                        break;
                }
                if (childidx == numchildren) return NULL;
            }
            it->node_child = childidx;
        }
        if (parent) *parent = p;
        return raxNodeFirstChildPtr(p)+childidx;
    }

    if (it->node == NULL) return NULL;
    if (it->node == it->rt->head) {
        if (parent) *parent = NULL;
        return &it->rt->head;
    }

    p = raxStackPeek(&it->stack);
    if (p == NULL) return NULL;
    cp = raxNodeFirstChildPtr(p);
    numchildren = p->iscompr ? 1 : p->size;
    childidx = it->node_child;

    if (childidx >= 0 && childidx < numchildren) {
        raxNode *child;
        memcpy(&child,cp+childidx,sizeof(child));
        if (child == it->node) {
            if (parent) *parent = p;
            return cp+childidx;
        }
    }

    for (childidx = 0; childidx < numchildren; childidx++) {
        raxNode *child;
        memcpy(&child,cp+childidx,sizeof(child));
        if (child == it->node) {
            it->node_child = childidx;
            if (parent) *parent = p;
            return cp+childidx;
        }
    }
    return NULL;
}

/* Do an iteration step towards the next element. At the end of the step the
 * iterator key will represent the (new) current key. If it is not possible
 * to step in the specified direction since there are no longer elements, the
 * iterator is flagged with RAX_ITER_EOF.
 *
 * If 'noup' is true the function starts directly scanning for the next
 * lexicographically smaller children, and the current node is already assumed
 * to be the parent of the last key node, so the first operation to go back to
 * the parent will be skipped. This option is used by raxSeek() when
 * implementing seeking a non existing element with the ">" or "<" options:
 * the starting node is not a key in that particular case, so we start the scan
 * from a node that does not represent the key set.
 *
 * The function returns 1 on success or 0 on out of memory. */
int raxIteratorNextStep(raxIterator *it, int noup) {
    if (it->flags & RAX_ITER_EOF) {
        return 1;
    } else if (it->flags & RAX_ITER_JUST_SEEKED) {
        it->flags &= ~RAX_ITER_JUST_SEEKED;
        return 1;
    }

    /* Save key len, stack items and the node where we are currently
     * so that on iterator EOF we can restore the current key and state. */
    size_t orig_key_len = it->key_len;
    size_t orig_stack_items = it->stack.items;
    raxNode *orig_node = it->node;
    int orig_flags = it->flags;
    void *orig_data = it->data;
    int orig_node_child = it->node_child;

    /* Inline leaves are represented by their parent node plus the current
     * key/data. They have no children, so the next step starts by going
     * "up" from the parent without popping the stack first. */
    if (raxIteratorIsInlineLeaf(it)) {
        raxIteratorClearInlineLeaf(it);
        noup = 1;
    }

    while(1) {
        int children = it->node->iscompr ? 1 : it->node->size;
        if (!noup && children) {
            debugf("GO DEEPER\n");
            /* Seek the lexicographically smaller key in this subtree, which
             * is the first one found always going towards the first child
             * of every successive node. */
            raxNode **cp = raxNodeFirstChildPtr(it->node);
            if (!raxIteratorEnterChild(it,it->node,cp,0)) return 0;
            /* For "next" step, stop every time we find a key along the
             * way, since the key is lexicographically smaller compared to
             * what follows in the sub-children. */
            if (raxIteratorIsInlineLeaf(it) || it->node->iskey) {
                return 1;
            }
        } else {
            /* If we finished exporing the previous sub-tree, switch to the
             * new one: go upper until a node is found where there are
             * children representing keys lexicographically greater than the
             * current key. */
            while(1) {
                int old_noup = noup;

                /* Already on head? Can't go up, iteration finished. */
                if (!noup && it->node == it->rt->head) {
                    it->flags = orig_flags | RAX_ITER_EOF;
                    it->stack.items = orig_stack_items;
                    it->key_len = orig_key_len;
                    it->node = orig_node;
                    it->data = orig_data;
                    it->node_child = orig_node_child;
                    return 1;
                }
                /* If there are no children at the current node, try parent's
                 * next child. */
                unsigned char prevchild = it->key[it->key_len-1];
                int prevchildidx = it->node_child;
                if (!noup) {
                    it->node = raxStackPop(&it->stack);
                    it->node_child = -1;
                } else {
                    noup = 0;
                }
                /* Adjust the current key to represent the node we are
                 * at. */
                int todel = it->node->iscompr ? it->node->size : 1;
                raxIteratorDelChars(it,todel);

                /* Try visiting the next child if there was at least one
                 * additional child. */
                if (!it->node->iscompr && it->node->size > (old_noup ? 0 : 1)) {
                    raxNode **cp = raxNodeFirstChildPtr(it->node);
                    int i;

                    if (prevchildidx != -1) {
                        i = prevchildidx+1;
                        cp += i;
                    } else {
                        i = 0;
                        while (i < it->node->size) {
                            debugf("SCAN NEXT %c\n", it->node->data[i]);
                            if (it->node->data[i] > prevchild) break;
                            i++;
                            cp++;
                        }
                    }
                    if (i != it->node->size) {
                        debugf("SCAN found a new node\n");
                        if (!raxIteratorEnterChild(it,it->node,cp,i))
                            return 0;
                        if (raxIteratorIsInlineLeaf(it) || it->node->iskey) {
                            return 1;
                        }
                        break;
                    }
                }
                if (old_noup) it->node_child = -1;
            }
        }
    }
}

/* Seek the greatest key in the subtree at the current node. Return 0 on
 * out of memory, otherwise 1. This is an helper function for different
 * iteration functions below. */
int raxSeekGreatest(raxIterator *it) {
    while(it->node->size) {
        raxNode **cp = raxNodeLastChildPtr(it->node);
        int lastidx = it->node->iscompr ? 0 : it->node->size-1;
        if (!raxIteratorEnterChild(it,it->node,cp,lastidx)) return 0;
        if (raxIteratorIsInlineLeaf(it)) return 1;
    }
    return 1;
}

/* Like raxIteratorNextStep() but implements an iteration step moving
 * to the lexicographically previous element. The 'noup' option has a similar
 * effect to the one of raxIteratorNextStep(). */
int raxIteratorPrevStep(raxIterator *it, int noup) {
    if (it->flags & RAX_ITER_EOF) {
        return 1;
    } else if (it->flags & RAX_ITER_JUST_SEEKED) {
        it->flags &= ~RAX_ITER_JUST_SEEKED;
        return 1;
    }

    /* Save key len, stack items and the node where we are currently
     * so that on iterator EOF we can restore the current key and state. */
    size_t orig_key_len = it->key_len;
    size_t orig_stack_items = it->stack.items;
    raxNode *orig_node = it->node;
    int orig_flags = it->flags;
    void *orig_data = it->data;
    int orig_node_child = it->node_child;

    if (raxIteratorIsInlineLeaf(it)) {
        raxIteratorClearInlineLeaf(it);
        noup = 1;
    }

    while(1) {
        int old_noup = noup;

        /* Already on head? Can't go up, iteration finished. */
        if (!noup && it->node == it->rt->head) {
            it->flags = orig_flags | RAX_ITER_EOF;
            it->stack.items = orig_stack_items;
            it->key_len = orig_key_len;
            it->node = orig_node;
            it->data = orig_data;
            it->node_child = orig_node_child;
            return 1;
        }

        unsigned char prevchild = it->key[it->key_len-1];
        int prevchildidx = it->node_child;
        if (!noup) {
            it->node = raxStackPop(&it->stack);
            it->node_child = -1;
        } else {
            noup = 0;
        }

        /* Adjust the current key to represent the node we are
         * at. */
        int todel = it->node->iscompr ? it->node->size : 1;
        raxIteratorDelChars(it,todel);

        /* Try visiting the prev child if there is at least one
         * child. */
        if (!it->node->iscompr && it->node->size > (old_noup ? 0 : 1)) {
            raxNode **cp;
            int i;

            if (prevchildidx != -1) {
                i = prevchildidx-1;
                cp = i == -1 ? NULL : raxNodeFirstChildPtr(it->node)+i;
            } else {
                cp = raxNodeLastChildPtr(it->node);
                i = it->node->size-1;
                while (i >= 0) {
                    debugf("SCAN PREV %c\n", it->node->data[i]);
                    if (it->node->data[i] < prevchild) break;
                    i--;
                    cp--;
                }
            }
            /* If we found a new subtree to explore in this node,
             * go deeper following all the last children in order to
             * find the key lexicographically greater. */
            if (i != -1) {
                debugf("SCAN found a new node\n");
                if (!raxIteratorEnterChild(it,it->node,cp,i)) return 0;
                if (raxIteratorIsInlineLeaf(it)) return 1;
                /* Seek sub-tree max. */
                if (!raxSeekGreatest(it)) return 0;
            }
        }
        if (old_noup) it->node_child = -1;

        /* Return the key: this could be the key we found scanning a new
         * subtree, or if we did not find a new subtree to explore here,
         * before giving up with this node, check if it's a key itself. */
        if (raxIteratorIsInlineLeaf(it) || it->node->iskey) {
            if (!raxIteratorIsInlineLeaf(it)) it->data = raxGetData(it->node);
            return 1;
        }
    }
}

/* Seek an iterator at the specified element.
 * Return 0 if the seek failed for syntax error or out of memory. Otherwise
 * 1 is returned. When 0 is returned for out of memory, errno is set to
 * the ENOMEM value. */
int raxSeek(raxIterator *it, const char *op, unsigned char *ele, size_t len) {
    int eq = 0, lt = 0, gt = 0, first = 0, last = 0;

    it->stack.items = 0; /* Just resetting. Initialized by raxStart(). */
    it->flags |= RAX_ITER_JUST_SEEKED;
    it->flags &= ~(RAX_ITER_EOF|RAX_ITER_INLINE_LEAF);
    it->key_len = 0;
    it->node = NULL;
    it->node_child = -1;

    /* Set flags according to the operator used to perform the seek. */
    if (op[0] == '>') {
        gt = 1;
        if (op[1] == '=') eq = 1;
    } else if (op[0] == '<') {
        lt = 1;
        if (op[1] == '=') eq = 1;
    } else if (op[0] == '=') {
        eq = 1;
    } else if (op[0] == '^') {
        first = 1;
    } else if (op[0] == '$') {
        last = 1;
    } else {
        errno = 0;
        return 0; /* Error. */
    }

    /* If there are no elements, set the EOF condition immediately and
     * return. */
    if (it->rt->numele == 0) {
        it->flags |= RAX_ITER_EOF;
        return 1;
    }

    if (first) {
        /* Seeking the first key greater or equal to the empty string
         * is equivalent to seeking the smaller key available. */
        return raxSeek(it,">=",NULL,0);
    }

    if (last) {
        /* Find the greatest key taking always the last child till a
         * final node is found. */
        it->node = it->rt->head;
        if (!raxSeekGreatest(it)) return 0;
        assert(raxIteratorIsInlineLeaf(it) || it->node->iskey);
        if (!raxIteratorIsInlineLeaf(it))
            it->data = raxGetData(it->node);
        return 1;
    }

    /* We need to seek the specified key. What we do here is to actually
     * perform a lookup, and later invoke the prev/next key code that
     * we already use for iteration. */
    int splitpos = 0;
    int inline_leaf = 0;
    raxNode **parentlink = NULL;
    size_t i = raxLowWalk(it->rt,ele,len,&it->node,&parentlink,&splitpos,&it->stack,&inline_leaf);

    /* Return OOM on incomplete stack info. */
    if (it->stack.oom) return 0;

    if (inline_leaf) {
        if (!raxIteratorAddChars(it,ele,i)) return 0;
        raxIteratorSetInlineLeaf(it,it->node,parentlink,
            (int)(parentlink - raxNodeFirstChildPtr(it->node)));

        if (eq && i == len) return 1;

        if (lt || gt) {
            it->flags &= ~RAX_ITER_JUST_SEEKED;
            if (i == len) {
                if (gt && !raxIteratorNextStep(it,0)) return 0;
                if (lt && !raxIteratorPrevStep(it,0)) return 0;
            } else if (gt) {
                if (!raxIteratorNextStep(it,0)) return 0;
            }
            it->flags |= RAX_ITER_JUST_SEEKED;
            return 1;
        }

        it->flags |= RAX_ITER_EOF;
        return 1;
    }

    if (eq && i == len && (!it->node->iscompr || splitpos == 0) &&
        it->node->iskey)
    {
        /* We found our node, since the key matches and we have an
         * "equal" condition. */
        if (!raxIteratorAddChars(it,ele,len)) return 0; /* OOM. */
        it->data = raxGetData(it->node);
    } else if (lt || gt) {
        /* Exact key not found or eq flag not set. We have to set as current
         * key the one represented by the node we stopped at, and perform
         * a next/prev operation to seek. To reconstruct the key at this node
         * we start from the parent and go to the current node, accumulating
         * the characters found along the way. */
        if (!raxStackPush(&it->stack,it->node)) return 0;
        for (size_t j = 1; j < it->stack.items; j++) {
            raxNode *parent = it->stack.stack[j-1];
            raxNode *child = it->stack.stack[j];
            if (parent->iscompr) {
                if (!raxIteratorAddChars(it,parent->data,parent->size))
                    return 0;
            } else {
                raxNode **cp = raxNodeFirstChildPtr(parent);
                unsigned char *p = parent->data;
                while(1) {
                    raxNode *aux;
                    memcpy(&aux,cp,sizeof(aux));
                    if (aux == child) break;
                    cp++;
                    p++;
                }
                if (!raxIteratorAddChars(it,p,1)) return 0;
            }
        }
        raxStackPop(&it->stack);

        /* We need to set the iterator in the correct state to call next/prev
         * step in order to seek the desired element. */
        debugf("After initial seek: i=%d len=%d key=%.*s\n",
            (int)i, (int)len, (int)it->key_len, it->key);
        if (i != len && !it->node->iscompr) {
            /* If we stopped in the middle of a normal node because of a
             * mismatch, add the mismatching character to the current key
             * and call the iterator with the 'noup' flag so that it will try
             * to seek the next/prev child in the current node directly based
             * on the mismatching character. */
            if (!raxIteratorAddChars(it,ele+i,1)) return 0;
            debugf("Seek normal node on mismatch: %.*s\n",
                (int)it->key_len, (char*)it->key);

            it->flags &= ~RAX_ITER_JUST_SEEKED;
            if (lt && !raxIteratorPrevStep(it,1)) return 0;
            if (gt && !raxIteratorNextStep(it,1)) return 0;
            it->flags |= RAX_ITER_JUST_SEEKED; /* Ignore next call. */
        } else if (i != len && it->node->iscompr) {
            debugf("Compressed mismatch: %.*s\n",
                (int)it->key_len, (char*)it->key);
            /* In case of a mismatch within a compressed node. */
            int nodechar = it->node->data[splitpos];
            int keychar = ele[i];
            it->flags &= ~RAX_ITER_JUST_SEEKED;
            if (gt) {
                /* If the key the compressed node represents is greater
                 * than our seek element, continue forward, otherwise set the
                 * state in order to go back to the next sub-tree. */
                if (nodechar > keychar) {
                    if (!raxIteratorNextStep(it,0)) return 0;
                } else {
                    if (!raxIteratorAddChars(it,it->node->data,it->node->size))
                        return 0;
                    if (!raxIteratorNextStep(it,1)) return 0;
                }
            }
            if (lt) {
                /* If the key the compressed node represents is smaller
                 * than our seek element, seek the greater key in this
                 * subtree, otherwise set the state in order to go back to
                 * the previous sub-tree. */
                if (nodechar < keychar) {
                    if (!raxSeekGreatest(it)) return 0;
                    if (!raxIteratorIsInlineLeaf(it))
                        it->data = raxGetData(it->node);
                } else {
                    if (!raxIteratorAddChars(it,it->node->data,it->node->size))
                        return 0;
                    if (!raxIteratorPrevStep(it,1)) return 0;
                }
            }
            it->flags |= RAX_ITER_JUST_SEEKED; /* Ignore next call. */
        } else {
            debugf("No mismatch: %.*s\n",
                (int)it->key_len, (char*)it->key);
            /* If there was no mismatch we are into a node representing the
             * key, (but which is not a key or the seek operator does not
             * include 'eq'), or we stopped in the middle of a compressed node
             * after processing all the key. Continue iterating as this was
             * a legitimate key we stopped at. */
            it->flags &= ~RAX_ITER_JUST_SEEKED;
            if (it->node->iscompr && it->node->iskey && splitpos && lt) {
                /* If we stopped in the middle of a compressed node with
                 * perfect match, and the condition is to seek a key "<" than
                 * the specified one, then if this node is a key it already
                 * represents our match. For instance we may have nodes:
                 *
                 * "f" -> "oobar" = 1 -> "" = 2
                 *
                 * Representing keys "f" = 1, "foobar" = 2. A seek for
                 * the key < "foo" will stop in the middle of the "oobar"
                 * node, but will be our match, representing the key "f".
                 *
                 * So in that case, we don't seek backward. */
                it->data = raxGetData(it->node);
            } else {
                if (gt && !raxIteratorNextStep(it,0)) return 0;
                if (lt && !raxIteratorPrevStep(it,0)) return 0;
            }
            it->flags |= RAX_ITER_JUST_SEEKED; /* Ignore next call. */
        }
    } else {
        /* If we are here just eq was set but no match was found. */
        it->flags |= RAX_ITER_EOF;
        return 1;
    }
    return 1;
}

/* Go to the next element in the scope of the iterator 'it'.
 * If EOF (or out of memory) is reached, 0 is returned, otherwise 1 is
 * returned. In case 0 is returned because of OOM, errno is set to ENOMEM. */
int raxNext(raxIterator *it) {
    if (!raxIteratorNextStep(it,0)) {
        errno = ENOMEM;
        return 0;
    }
    if (it->flags & RAX_ITER_EOF) {
        errno = 0;
        return 0;
    }
    return 1;
}

/* Go to the previous element in the scope of the iterator 'it'.
 * If EOF (or out of memory) is reached, 0 is returned, otherwise 1 is
 * returned. In case 0 is returned because of OOM, errno is set to ENOMEM. */
int raxPrev(raxIterator *it) {
    if (!raxIteratorPrevStep(it,0)) {
        errno = ENOMEM;
        return 0;
    }
    if (it->flags & RAX_ITER_EOF) {
        errno = 0;
        return 0;
    }
    return 1;
}

/* Perform a random walk starting in the current position of the iterator.
 * Return 0 if the tree is empty or on out of memory. Otherwise 1 is returned
 * and the iterator is set to the node reached after doing a random walk
 * of 'steps' steps. If the 'steps' argument is 0, the random walk is performed
 * using a random number of steps between 1 and two times the logarithm of
 * the number of elements.
 *
 * NOTE: if you use this function to generate random elements from the radix
 * tree, expect a disappointing distribution. A random walk produces good
 * random elements if the tree is not sparse, however in the case of a radix
 * tree certain keys will be reported much more often than others. At least
 * this function should be able to expore every possible element eventually. */
int raxRandomWalk(raxIterator *it, size_t steps) {
    if (it->rt->numele == 0) {
        it->flags |= RAX_ITER_EOF;
        return 0;
    }

    if (steps == 0) {
        size_t fle = 1+floor(log(it->rt->numele));
        fle *= 2;
        steps = 1 + rand() % fle;
    }

    raxNode *n = it->node;
    int inline_leaf = raxIteratorIsInlineLeaf(it);
    int node_child = it->node_child;
    while(steps > 0 || !(inline_leaf || n->iskey)) {
        if (inline_leaf) {
            inline_leaf = 0;
            int todel = n->iscompr ? n->size : 1;
            raxIteratorDelChars(it,todel);
            if (n->iskey) steps--;
            continue;
        }

        int numchildren = n->iscompr ? 1 : n->size;
        int r = rand() % (numchildren+(n != it->rt->head));

        if (r == numchildren) {
            /* Go up to parent. */
            n = raxStackPop(&it->stack);
            node_child = -1;
            int todel = n->iscompr ? n->size : 1;
            raxIteratorDelChars(it,todel);
        } else {
            /* Select a random child. */
            raxNode **cp = raxNodeFirstChildPtr(n)+r;
            int cidx = n->iscompr ? 0 : r;
            if (n->iscompr) {
                if (!raxIteratorAddChars(it,n->data,n->size)) return 0;
            } else {
                if (!raxIteratorAddChars(it,n->data+r,1)) return 0;
            }
            if (raxIsInlineLeaf(n,cidx)) {
                memcpy(&it->data,cp,sizeof(it->data));
                inline_leaf = 1;
                node_child = cidx;
            } else {
                if (!raxStackPush(&it->stack,n)) return 0;
                memcpy(&n,cp,sizeof(n));
                node_child = cidx;
            }
        }
        if (inline_leaf || n->iskey) steps--;
    }
    it->node = n;
    it->node_child = node_child;
    if (inline_leaf) {
        it->flags |= RAX_ITER_INLINE_LEAF;
    } else {
        it->flags &= ~RAX_ITER_INLINE_LEAF;
        it->data = raxGetData(it->node);
    }
    return 1;
}

/* Compare the key currently pointed by the iterator to the specified
 * key according to the specified operator. Returns 1 if the comparison is
 * true, otherwise 0 is returned. */
int raxCompare(raxIterator *iter, const char *op, unsigned char *key, size_t key_len) {
    int eq = 0, lt = 0, gt = 0;

    if (op[0] == '=' || op[1] == '=') eq = 1;
    if (op[0] == '>') gt = 1;
    else if (op[0] == '<') lt = 1;
    else if (op[1] != '=') return 0; /* Syntax error. */

    size_t minlen = key_len < iter->key_len ? key_len : iter->key_len;
    int cmp = memcmp(iter->key,key,minlen);

    /* Handle == */
    if (lt == 0 && gt == 0) return cmp == 0 && key_len == iter->key_len;

    /* Handle >, >=, <, <= */
    if (cmp == 0) {
        /* Same prefix: longer wins. */
        if (eq && key_len == iter->key_len) return 1;
        else if (lt) return iter->key_len < key_len;
        else if (gt) return iter->key_len > key_len;
        else return 0; /* Avoid warning, just 'eq' is handled before. */
    } else if (cmp > 0) {
        return gt ? 1 : 0;
    } else /* (cmp < 0) */ {
        return lt ? 1 : 0;
    }
}

/* Update the data associated to the element currently selected by the
 * iterator. The operation works both for regular key nodes and for inline
 * leaves represented virtually by the iterator.
 *
 * The function returns 1 on success or 0 on error. In case the iterator
 * is not positioned on an element errno is set to ENOENT. If the current
 * node needs to be reallocated in order to store a non-NULL value and the
 * allocation fails, errno is set to ENOMEM. */
int raxIteratorSetData(raxIterator *it, void *data) {
    raxNode **parentlink;

    if (it->rt == NULL || (it->flags & RAX_ITER_EOF)) {
        errno = ENOENT;
        return 0;
    }

    parentlink = raxIteratorCurrentParentLink(it,NULL);
    if (parentlink == NULL) {
        errno = ENOENT;
        return 0;
    }

    if (raxIteratorIsInlineLeaf(it)) {
        memcpy(parentlink,&data,sizeof(data));
        it->data = data;
        return 1;
    }

    if (it->node == NULL || !it->node->iskey) {
        errno = ENOENT;
        return 0;
    }

    if (it->node->isnull && data != NULL) {
        raxNode *newnode = raxReallocForData(it->node,data);
        if (newnode == NULL) {
            errno = ENOMEM;
            return 0;
        }
        if (newnode != it->node) {
            memcpy(parentlink,&newnode,sizeof(newnode));
            it->node = newnode;
        }
    }

    raxSetData(it->node,data);
    it->data = data;
    return 1;
}

/* ----------------------- Defragmentation iterator -------------------------
 * The defragmentation iterator scans the radix tree structure itself and
 * yields either real raxNode allocations or non-NULL values associated with
 * keys. It is designed to relocate nodes and values while keeping the tree
 * and the iterator state valid.
 * ----------------------------------------------------------------------- */

#define RAX_DEFRAG_STATE_EMIT_NODE 0
#define RAX_DEFRAG_STATE_EMIT_DATA 1
#define RAX_DEFRAG_STATE_CHILDREN 2

/* Initialize the private DFS stack used by the defragmentation iterator. */
static inline void raxDefragStackInit(raxDefragIterator *it) {
    it->stack = it->static_items;
    it->items = 0;
    it->maxitems = RAX_DEFRAG_STATIC_ITEMS;
}

/* Push a new node into the defragmentation iterator stack. The parent_child
 * argument is the child index of this node in its parent, or -1 for the
 * radix tree head. Return 1 on success or 0 on out of memory. */
static inline int raxDefragStackPush(raxDefragIterator *it, raxNode *node,
                                     int parent_child)
{
    if (it->items == it->maxitems) {
        if (it->stack == it->static_items) {
            it->stack = rax_malloc(sizeof(*it->stack)*it->maxitems*2);
            if (it->stack == NULL) {
                it->stack = it->static_items;
                errno = ENOMEM;
                return 0;
            }
            memcpy(it->stack,it->static_items,sizeof(*it->stack)*it->maxitems);
        } else {
            raxDefragFrame *newalloc =
                rax_realloc(it->stack,sizeof(*it->stack)*it->maxitems*2);
            if (newalloc == NULL) {
                errno = ENOMEM;
                return 0;
            }
            it->stack = newalloc;
        }
        it->maxitems *= 2;
    }

    it->stack[it->items].node = node;
    it->stack[it->items].child = 0;
    it->stack[it->items].parent_child = parent_child;
    it->stack[it->items].state = RAX_DEFRAG_STATE_EMIT_NODE;
    it->items++;
    return 1;
}

/* Return the frame at the top of the defragmentation stack, or NULL if there
 * are no more nodes to visit. */
static inline raxDefragFrame *raxDefragStackPeek(raxDefragIterator *it) {
    if (it->items == 0) return NULL;
    return &it->stack[it->items-1];
}

/* Pop the current frame from the defragmentation stack. */
static inline void raxDefragStackPop(raxDefragIterator *it) {
    if (it->items) it->items--;
}

/* Free the stack allocation if the iterator used heap memory. */
static inline void raxDefragStackFree(raxDefragIterator *it) {
    if (it->stack != it->static_items) rax_free(it->stack);
}

/* Append characters at the current key string of the defragmentation
 * iterator. Like the normal iterator, the key is rebuilt incrementally as
 * the walk descends and climbs the radix tree. */
static int raxDefragAddChars(raxDefragIterator *it, unsigned char *s,
                             size_t len)
{
    if (len == 0) return 1;
    if (it->key_max < it->key_len+len) {
        int from_static = it->key == it->key_static_string;
        unsigned char *old = from_static ? NULL : it->key;
        size_t new_max = (it->key_len+len)*2;
        unsigned char *new_key = rax_realloc(old,new_max);
        if (new_key == NULL) {
            errno = ENOMEM;
            return 0;
        }
        it->key = new_key;
        if (from_static) memcpy(it->key,it->key_static_string,it->key_len);
        it->key_max = new_max;
    }
    memcpy(it->key+it->key_len,s,len);
    it->key_len += len;
    return 1;
}

/* Remove the specified number of chars from the right of the current key. */
static inline void raxDefragDelChars(raxDefragIterator *it, size_t count) {
    it->key_len -= count;
}

/* Build the flags associated with the current item returned by the
 * defragmentation iterator. */
static inline int raxDefragNodeFlags(raxDefragIterator *it, raxNode *node) {
    int flags = 0;
    if (it->items == 1) flags |= RAX_DEFRAG_F_ROOT;
    if (node->iskey) flags |= RAX_DEFRAG_F_KEY;
    if (node->iscompr) flags |= RAX_DEFRAG_F_COMPRESSED;
    return flags;
}

/* Initialize a defragmentation iterator. Unlike the normal iterator there
 * is no need to seek: the iterator performs a full traversal from the root. */
void raxDefragStart(raxDefragIterator *it, rax *rt) {
    it->kind = 0;
    it->flags = 0;
    it->rt = rt;
    it->key_len = 0;
    it->key = it->key_static_string;
    it->key_max = RAX_ITER_STATIC_LEN;
    it->size = 0;
    it->ptr = NULL;
    it->node = NULL;
    it->node_child = -1;
    it->pending_todel = 0;
    it->eof = (rt == NULL || rt->head == NULL);
    raxDefragStackInit(it);
    if (!it->eof) raxDefragStackPush(it,rt->head,-1);
}

/* Return the next node or data pointer in the defragmentation walk.
 * Return 1 if a new item was returned, 0 if the scan is finished or if
 * an out of memory error happened while extending the iterator state.
 *
 * The walk is preorder over the real nodes of the radix tree. Every node is
 * returned first as a NODE item. If the node also represents a key with
 * non-NULL associated data, the same node is returned again as a DATA item.
 * Inline leaves are returned only as DATA items, since there is no standalone
 * node allocation to relocate for them. */
int raxDefragNext(raxDefragIterator *it) {
    raxDefragFrame *frame;

    if (it->eof) return 0;
    if (it->pending_todel) {
        raxDefragDelChars(it,it->pending_todel);
        it->pending_todel = 0;
    }

    while((frame = raxDefragStackPeek(it)) != NULL) {
        raxNode *node = frame->node;
        int numchildren = node->iscompr ? 1 : node->size;

        if (frame->state == RAX_DEFRAG_STATE_EMIT_NODE) {
            frame->state = RAX_DEFRAG_STATE_EMIT_DATA;
            it->kind = RAX_DEFRAG_NODE;
            it->flags = raxDefragNodeFlags(it,node);
            it->size = raxNodeCurrentLength(node);
            it->ptr = node;
            it->node = node;
            it->node_child = frame->parent_child;
            return 1;
        } else if (frame->state == RAX_DEFRAG_STATE_EMIT_DATA) {
            frame->state = RAX_DEFRAG_STATE_CHILDREN;
            if (node->iskey && !node->isnull) {
                it->kind = RAX_DEFRAG_DATA;
                it->flags = raxDefragNodeFlags(it,node);
                it->size = 0;
                it->ptr = raxGetData(node);
                it->node = node;
                it->node_child = frame->parent_child;
                return 1;
            }
        } else {
            if (frame->child == (size_t)numchildren) {
                raxDefragStackPop(it);
                if (it->items == 0) {
                    it->eof = 1;
                    return 0;
                }
                raxNode *parent = raxDefragStackPeek(it)->node;
                raxDefragDelChars(it,parent->iscompr ? parent->size : 1);
                continue;
            }

            int childidx = frame->child++;
            raxNode **childfield = raxNodeFirstChildPtr(node)+childidx;
            size_t addlen = node->iscompr ? node->size : 1;
            unsigned char *s = node->iscompr ? node->data : node->data+childidx;

            if (raxIsInlineLeaf(node,childidx)) {
                void *value;
                memcpy(&value,childfield,sizeof(value));
                if (value == NULL) continue;
                if (!raxDefragAddChars(it,s,addlen)) return 0;
                it->pending_todel = addlen;
                it->kind = RAX_DEFRAG_DATA;
                it->flags = RAX_DEFRAG_F_KEY|RAX_DEFRAG_F_INLINE_DATA|
                            (node->iscompr ? RAX_DEFRAG_F_COMPRESSED : 0);
                it->size = 0;
                it->ptr = value;
                it->node = node;
                it->node_child = childidx;
                return 1;
            } else {
                raxNode *child;
                memcpy(&child,childfield,sizeof(child));
                if (!raxDefragAddChars(it,s,addlen)) return 0;
                if (!raxDefragStackPush(it,child,childidx)) {
                    raxDefragDelChars(it,addlen);
                    return 0;
                }
            }
        }
    }

    it->eof = 1;
    return 0;
}

/* Replace the current NODE item with 'newptr', returning the old node pointer
 * on success or NULL on error. The caller is responsible for allocating and
 * copying the new node before calling this function.
 *
 * The function updates the radix tree parent link, or the tree head if the
 * current node is the root, and also updates the iterator internal state so
 * that the defragmentation walk can continue using the new node. */
void *raxDefragReplaceNode(raxDefragIterator *it, void *newptr) {
    raxNode *oldnode, *newnode = newptr;
    raxDefragFrame *frame;

    if (it->eof) {
        errno = ENOENT;
        return NULL;
    }
    if (it->kind != RAX_DEFRAG_NODE || newptr == NULL) {
        errno = EINVAL;
        return NULL;
    }

    frame = raxDefragStackPeek(it);
    if (frame == NULL || frame->node != it->node) {
        errno = ENOENT;
        return NULL;
    }

    oldnode = it->node;
    if (it->items == 1) {
        it->rt->head = newnode;
    } else {
        raxDefragFrame *parent = &it->stack[it->items-2];
        raxNode **childfield =
            raxNodeFirstChildPtr(parent->node)+frame->parent_child;
        memcpy(childfield,&newnode,sizeof(newnode));
    }
    frame->node = newnode;
    it->node = newnode;
    it->ptr = newnode;
    return oldnode;
}

/* Replace the current DATA item with 'newptr', returning the old data pointer
 * on success or NULL on error. Inline leaves are updated in place without
 * materializing a real node.
 *
 * Since DATA items are returned only for non-NULL values, the function only
 * needs to patch the stored pointer and does not need to change the node
 * layout. */
void *raxDefragReplaceData(raxDefragIterator *it, void *newptr) {
    void *oldptr;

    if (it->eof) {
        errno = ENOENT;
        return NULL;
    }
    if (it->kind != RAX_DEFRAG_DATA || it->node == NULL) {
        errno = EINVAL;
        return NULL;
    }

    oldptr = it->ptr;
    if (it->flags & RAX_DEFRAG_F_INLINE_DATA) {
        raxNode **childfield = raxNodeFirstChildPtr(it->node)+it->node_child;
        memcpy(childfield,&newptr,sizeof(newptr));
    } else {
        raxSetData(it->node,newptr);
    }
    it->ptr = newptr;
    return oldptr;
}

/* Free the defragmentation iterator. */
void raxDefragStop(raxDefragIterator *it) {
    if (it->key != it->key_static_string) rax_free(it->key);
    raxDefragStackFree(it);
}

/* Free the iterator. */
void raxStop(raxIterator *it) {
    if (it->key != it->key_static_string) rax_free(it->key);
    raxStackFree(&it->stack);
}

/* Return if the iterator is in an EOF state. This happens when raxSeek()
 * failed to seek an appropriate element, so that raxNext() or raxPrev()
 * will return zero, or when an EOF condition was reached while iterating
 * with raxNext() and raxPrev(). */
int raxEOF(raxIterator *it) {
    return it->flags & RAX_ITER_EOF;
}

/* Return the number of elements inside the radix tree. */
uint64_t raxSize(rax *rax) {
    return rax->numele;
}

/* ----------------------------- Introspection ------------------------------ */

/* This function is mostly used for debugging and learning purposes.
 * It shows an ASCII representation of a tree on standard output, outling
 * all the nodes and the contained keys.
 *
 * The representation is as follow:
 *
 *  "foobar" (compressed node)
 *  [abc] (normal node with three children)
 *  [abc]=0x12345678 (node is a key, pointing to value 0x12345678)
 *  [] (a normal empty node)
 *
 *  Children are represented in new idented lines, each children prefixed by
 *  the "`-(x)" string, where "x" is the edge byte.
 *
 *  [abc]
 *   `-(a) "ladin"
 *   `-(b) [kj]
 *   `-(c) []
 *
 *  However when a node has a single child the following representation
 *  is used instead:
 *
 *  [abc] -> "ladin" -> []
 */

/* The actual implementation of raxShow(). */
void raxRecursiveShow(int level, int lpad, raxNode *n) {
    char s = n->iscompr ? '"' : '[';
    char e = n->iscompr ? '"' : ']';

    int numchars = printf("%c%.*s%c", s, n->size, n->data, e);
    if (n->iskey) {
        numchars += printf("=%p",raxGetData(n));
    }

    int numchildren = n->iscompr ? 1 : n->size;
    /* Note that 7 and 4 magic constants are the string length
     * of " `-(x) " and " -> " respectively. */
    if (level) {
        lpad += (numchildren > 1) ? 7 : 4;
        if (numchildren == 1) lpad += numchars;
    }
    raxNode **cp = raxNodeFirstChildPtr(n);
    for (int i = 0; i < numchildren; i++) {
        char *branch = " `-(%c) ";
        if (numchildren > 1) {
            printf("\n");
            for (int j = 0; j < lpad; j++) putchar(' ');
            printf(branch,n->data[i]);
        } else {
            printf(" -> ");
        }
        if (raxIsInlineLeaf(n,i)) {
            void *val;
            memcpy(&val,cp,sizeof(val));
            printf("[]=%p",val);
        } else {
            raxNode *child;
            memcpy(&child,cp,sizeof(child));
            raxRecursiveShow(level+1,lpad,child);
        }
        cp++;
    }
}

/* Show a tree, as outlined in the comment above. */
void raxShow(rax *rax) {
    raxRecursiveShow(0,0,rax->head);
    putchar('\n');
}

/* Used by debugnode() macro to show info about a given node. */
void raxDebugShowNode(const char *msg, raxNode *n) {
    if (raxDebugMsg == 0) return;
    printf("%s: %p [%.*s] key:%d size:%d children:",
        msg, (void*)n, (int)n->size, (char*)n->data, n->iskey, n->size);
    int numcld = n->iscompr ? 1 : n->size;
    raxNode **cldptr = raxNodeLastChildPtr(n) - (numcld-1);
    while(numcld--) {
        raxNode *child;
        memcpy(&child,cldptr,sizeof(child));
        cldptr++;
        printf("%p ", (void*)child);
    }
    printf("\n");
    fflush(stdout);
}

/* Touch all the nodes of a tree returning a check sum. This is useful
 * in order to make Valgrind detect if there is something wrong while
 * reading the data structure.
 *
 * This function was used in order to identify Rax bugs after a big refactoring
 * using this technique:
 *
 * 1. The rax-test is executed using Valgrind, adding a printf() so that for
 *    the fuzz tester we see what iteration in the loop we are in.
 * 2. After every modification of the radix tree made by the fuzz tester
 *    in rax-test.c, we add a call to raxTouch().
 * 3. Now as soon as an operation will corrupt the tree, raxTouch() will
 *    detect it (via Valgrind) immediately. We can add more calls to narrow
 *    the state.
 * 4. At this point a good idea is to enable Rax debugging messages immediately
 *    before the moment the tree is corrupted, to see what happens.
 */
unsigned long raxTouch(raxNode *n) {
    debugf("Touching %p\n", (void*)n);
    unsigned long sum = 0;
    if (n->iskey) {
        sum += (unsigned long)raxGetData(n);
    }

    int numchildren = n->iscompr ? 1 : n->size;
    raxNode **cp = raxNodeFirstChildPtr(n);
    int count = 0;
    for (int i = 0; i < numchildren; i++) {
        if (numchildren > 1) {
            sum += (long)n->data[i];
        }
        if (raxIsInlineLeaf(n,i)) {
            void *val;
            memcpy(&val,cp,sizeof(val));
            sum += (unsigned long)val;
        } else {
            raxNode *child;
            memcpy(&child,cp,sizeof(child));
            if (child == (void*)0x65d1760) count++;
            if (count > 1) exit(1);
            sum += raxTouch(child);
        }
        cp++;
    }
    return sum;
}
