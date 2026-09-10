#include <stddef.h>
#include "rax.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ITEMS=256, CAP=96 };
static uint32_t seed=0x835790abu;
static uint32_t draw(void) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed; }
static unsigned char keys[ITEMS][CAP];
static size_t lens[ITEMS];
static int active[ITEMS];
static void *values[ITEMS];
static void insert(rax *tree, unsigned id) {
    for (unsigned j=0;j<ITEMS;j++)
        if(active[j] && lens[j]==lens[id] && !memcmp(keys[j],keys[id],lens[id])) active[j]=0;
    raxInsert(tree,keys[id],lens[id],values[id],NULL);
    active[id]=1;
}
int main(void) {
    rax *tree=raxNew();
    if(!tree) return 2;
    for(unsigned i=0;i<ITEMS;i++) {
        lens[i]=i<64?i:1+draw()%(CAP-16);
        for(size_t j=0;j<lens[i];j++) keys[i][j]=i<64?(unsigned char)('a'+j%3):(unsigned char)draw();
        values[i]=i%7?(void *)(uintptr_t)(0x1000u+i*16u):NULL;
        insert(tree,i);
    }
    for(unsigned step=0;step<30000;step++) {
        if(step%3==0) {
            unsigned id=draw()%ITEMS;
            if(active[id] && (draw()&1)) {
                if(!raxRemove(tree,keys[id],lens[id],NULL)) return 3;
                active[id]=0;
            } else insert(tree,id);
        }
        unsigned char query[CAP];
        unsigned id=draw()%ITEMS;
        size_t n;
        if(step%3) {
            n=lens[id];memcpy(query,keys[id],n);
            unsigned extra=draw()%12;
            for(unsigned j=0;j<extra;j++) query[n++]=(unsigned char)draw();
            if(step%5==0 && n) n=draw()%n;
        } else {
            n=draw()%CAP;
            for(size_t j=0;j<n;j++) query[j]=(unsigned char)draw();
        }
        size_t want_len=0,got_len=0,examined=0;
        void *want=raxNotFound;
        for(unsigned j=0;j<ITEMS;j++)
            if(active[j] && lens[j]<=n && !memcmp(keys[j],query,lens[j]) &&
               (want==raxNotFound || lens[j]>want_len)) {
                want=values[j];want_len=lens[j];
            }
        void *got=raxFindLongestPrefix(tree,query,n,&got_len,&examined);
        if(got!=want || got_len!=want_len) {
            fprintf(stderr,"prefix mismatch step=%u bytes=%zu expected=%zu observed=%zu\n",
                    step,n,want_len,got_len);
            return 1;
        }
    }
    raxFree(tree);
    puts("Rax longest-prefix oracle PASS: 30000 queries with insertion/removal, binary keys, empty keys, null values, and nested prefixes");
    return 0;
}
