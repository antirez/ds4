#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "../ds4_agent_git.h"

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void test_fail(const char *msg) {
    fprintf(stderr, "ds4_agent_git_test: %s\n", msg);
    exit(1);
}

#define CHECK(cond, msg) do { if (!(cond)) test_fail(msg); } while (0)

static char *test_strdup(const char *s) {
    size_t n = strlen(s);
    char *out = malloc(n + 1);
    CHECK(out != NULL, "malloc failed");
    memcpy(out, s, n + 1);
    return out;
}

static char *make_temp_dir(void) {
    const char *base = getenv("TMPDIR");
    if (!base || !base[0]) base = "/tmp";
    for (int i = 0; i < 100; i++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/ds4-agent-git-test-%ld-%ld-%d",
                 base, (long)getpid(), (long)time(NULL), i);
        if (mkdir(path, 0700) == 0) return test_strdup(path);
        if (errno != EEXIST) break;
    }
    test_fail("failed to create temp dir");
    return NULL;
}

static void run_cmd(char *const argv[]) {
    pid_t pid = fork();
    CHECK(pid >= 0, "fork failed");
    if (pid == 0) {
        execvp(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        test_fail("waitpid failed");
    }
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "command failed");
}

static void write_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL, "failed to open file");
    CHECK(fwrite(text, 1, strlen(text), fp) == strlen(text), "failed to write file");
    CHECK(fclose(fp) == 0, "failed to close file");
}

static char *join_path(const char *dir, const char *file) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    return test_strdup(path);
}

static void remove_tree(const char *path) {
    struct stat st;
    if (lstat(path, &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (d) {
            struct dirent *de;
            while ((de = readdir(d)) != NULL) {
                if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
                char *child = join_path(path, de->d_name);
                remove_tree(child);
                free(child);
            }
            closedir(d);
        }
        rmdir(path);
    } else {
        unlink(path);
    }
}

static void assert_git_ok(const char *repo, const char *action,
                          const char *path, const char *ref,
                          int limit, bool staged,
                          ds4_agent_git_result *r) {
    char err[256] = {0};
    CHECK(ds4_agent_git_run(repo, action, path, ref, limit, staged,
                            64 * 1024, r, err, sizeof(err)),
          "git helper failed to run");
    CHECK(r->exit_code == 0, "git command returned non-zero");
}

static void assert_git_opts_ok(const ds4_agent_git_options *opts,
                               ds4_agent_git_result *r) {
    ds4_agent_git_options copy = *opts;
    if (!copy.max_bytes) copy.max_bytes = 64 * 1024;
    char err[256] = {0};
    CHECK(ds4_agent_git_run_options(&copy, r, err, sizeof(err)),
          "git helper options failed to run");
    CHECK(r->exit_code == 0, "git options command returned non-zero");
}

int main(void) {
    char *repo = make_temp_dir();
    char *remote_dir = make_temp_dir();
    char *file = join_path(repo, "a.txt");
    char *untracked = join_path(repo, "b.txt");
    char *stash_file = join_path(repo, "c.txt");
    char *merge_file = join_path(repo, "d.txt");
    char *topic_file = join_path(repo, "e.txt");
    char *base_file = join_path(repo, "f.txt");

    char *init_argv[] = {"git", "-C", repo, "init", "-q", NULL};
    char *config_name_argv[] = {"git", "-C", repo, "config", "user.name", "DS4 Test", NULL};
    char *config_email_argv[] = {"git", "-C", repo, "config", "user.email", "ds4@example.invalid", NULL};
    char *bare_init_argv[] = {"git", "-C", remote_dir, "init", "--bare", "-q", NULL};
    run_cmd(init_argv);
    run_cmd(config_name_argv);
    run_cmd(config_email_argv);
    run_cmd(bare_init_argv);

    write_file(file, "one\n");
    char *add_argv[] = {"git", "-C", repo, "add", "a.txt", NULL};
    char *commit_argv[] = {"git", "-C", repo, "commit", "-q", "-m", "initial", NULL};
    run_cmd(add_argv);
    run_cmd(commit_argv);

    write_file(file, "two\n");
    write_file(untracked, "new\n");

    ds4_agent_git_result r = {0};
    char err[256] = {0};
    ds4_agent_git_options opts = {.repo = repo, .action = "info"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "repo_root=") != NULL, "info missing repo root");
    CHECK(strstr(r.output, "branch=") != NULL, "info missing branch");
    CHECK(strstr(r.output, "head=") != NULL, "info missing head");
    CHECK(strstr(r.output, "dirty=true") != NULL, "info missing dirty state");
    CHECK(strstr(r.output, "a.txt") != NULL, "info status missing modified file");
    ds4_agent_git_result_free(&r);

    assert_git_ok(repo, "status", NULL, NULL, 0, false, &r);
    CHECK(strstr(r.output, "a.txt") != NULL, "status missing modified file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "changed_files"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, " M a.txt") != NULL, "changed_files missing modified file");
    CHECK(strstr(r.output, "?? b.txt") != NULL, "changed_files missing untracked file");
    ds4_agent_git_result_free(&r);

    assert_git_ok(repo, "diff", "a.txt", NULL, 0, false, &r);
    CHECK(strstr(r.output, "-one") != NULL, "diff missing removed line");
    CHECK(strstr(r.output, "+two") != NULL, "diff missing added line");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .path = "a.txt",
        .name_status = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "M\ta.txt") != NULL, "name-status diff missing modified file");
    CHECK(strstr(r.output, "-one") == NULL, "name-status diff unexpectedly included patch");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .path = "a.txt",
        .stat = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "a.txt") != NULL, "stat diff missing modified file");
    CHECK(strstr(r.output, "-one") == NULL, "stat diff unexpectedly included patch");
    ds4_agent_git_result_free(&r);

    assert_git_ok(repo, "ls_files", NULL, NULL, 0, false, &r);
    CHECK(strstr(r.output, "a.txt") != NULL, "ls_files missing tracked file");
    ds4_agent_git_result_free(&r);

    assert_git_ok(repo, "log", NULL, NULL, 1, false, &r);
    CHECK(strstr(r.output, "initial") != NULL, "log missing commit subject");
    ds4_agent_git_result_free(&r);

    assert_git_ok(repo, "show", NULL, "HEAD", 0, false, &r);
    CHECK(strstr(r.output, "initial") != NULL, "show missing commit subject");
    CHECK(strstr(r.output, "a.txt") != NULL, "show missing file stat");
    ds4_agent_git_result_free(&r);

    run_cmd(add_argv);
    assert_git_ok(repo, "diff", "a.txt", NULL, 0, true, &r);
    CHECK(strstr(r.output, "-one") != NULL, "staged diff missing removed line");
    CHECK(strstr(r.output, "+two") != NULL, "staged diff missing added line");
    ds4_agent_git_result_free(&r);

    char *commit_update_argv[] = {"git", "-C", repo, "commit", "-q", "-m", "update a", NULL};
    run_cmd(commit_update_argv);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "show",
        .ref = "HEAD",
        .patch = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "update a") != NULL, "show patch missing commit subject");
    CHECK(strstr(r.output, "-one") != NULL, "show patch missing removed line");
    CHECK(strstr(r.output, "+two") != NULL, "show patch missing added line");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .range = "HEAD~1...HEAD",
        .name_status = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "M\ta.txt") != NULL, "range diff missing modified file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .base_ref = "HEAD~1",
        .target_ref = "HEAD",
        .name_only = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "a.txt") != NULL, "base/target diff missing file");
    CHECK(strstr(r.output, "-one") == NULL, "name-only diff unexpectedly included patch");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "file_at_ref",
        .ref = "HEAD~1",
        .path = "a.txt",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(!strcmp(r.output, "one\n"), "file_at_ref returned wrong historical content");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "blame",
        .ref = "HEAD",
        .path = "a.txt",
        .start_line = 1,
        .line_count = 1,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "two") != NULL, "blame missing requested line");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "path_history",
        .path = "a.txt",
        .limit = 5,
        .follow = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "update a") != NULL, "path_history missing update commit");
    CHECK(strstr(r.output, "initial") != NULL, "path_history missing initial commit");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stage",
        .path = "b.txt",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "?? b.txt") != NULL, "stage dry-run missing candidate file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .path = "b.txt",
        .staged = true,
        .name_only = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "b.txt") == NULL, "stage dry-run unexpectedly staged file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stage",
        .path = "b.txt",
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .path = "b.txt",
        .staged = true,
        .name_status = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "A\tb.txt") != NULL, "stage did not stage new file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "unstage",
        .path = "b.txt",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "A\tb.txt") != NULL, "unstage dry-run missing staged file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "unstage",
        .path = "b.txt",
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "changed_files", .path = "b.txt"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "?? b.txt") != NULL, "unstage did not unstage file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "stage", .path = "b.txt"};
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "commit",
        .message = "add b",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "b.txt") != NULL, "commit dry-run missing staged summary");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "commit",
        .message = "add b",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "add b") != NULL, "commit missing commit subject");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "log", .limit = 1};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "add b") != NULL, "log missing new commit");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "file_at_ref",
        .ref = "HEAD",
        .path = "b.txt",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(!strcmp(r.output, "new\n"), "commit did not preserve staged file content");
    ds4_agent_git_result_free(&r);

    write_file(file, "three\n");
    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "worktree_restore",
        .path = "a.txt",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "M\ta.txt") != NULL, "worktree_restore dry-run missing modified file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "worktree_restore", .path = "a.txt"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "worktree_restore should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "worktree_restore",
        .path = "a.txt",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "changed_files",
        .path = "a.txt",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "a.txt") == NULL, "worktree_restore left tracked file dirty");
    ds4_agent_git_result_free(&r);

    char *branch_argv[] = {"git", "-C", repo, "branch", "side", NULL};
    run_cmd(branch_argv);
    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "switch",
        .ref = "side",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strlen(r.output) >= 40, "switch dry-run did not resolve target");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "switch", .ref = "side"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "switch should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "switch",
        .ref = "side",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "status"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "## side") != NULL, "switch did not move to side branch");
    ds4_agent_git_result_free(&r);

    char *remote_add_argv[] = {"git", "-C", repo, "remote", "add", "local", remote_dir, NULL};
    run_cmd(remote_add_argv);
    opts = (ds4_agent_git_options){.repo = repo, .action = "remote_list"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "local") != NULL, "remote_list missing remote name");
    CHECK(strstr(r.output, remote_dir) != NULL, "remote_list missing remote path");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "push",
        .remote = "local",
        .ref = "side",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "push",
        .remote = "local",
        .ref = "side",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "push should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "push",
        .remote = "local",
        .ref = "side",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "side") != NULL, "push missing pushed ref");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "fetch",
        .remote = "local",
        .ref = "side",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "fetch",
        .remote = "local",
        .ref = "side",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "fetch should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "fetch",
        .remote = "local",
        .ref = "side",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    char *merge_branch_argv[] = {"git", "-C", repo, "switch", "-q", "-c", "merge_src", NULL};
    run_cmd(merge_branch_argv);
    write_file(merge_file, "merge\n");
    char *add_merge_argv[] = {"git", "-C", repo, "add", "d.txt", NULL};
    char *commit_merge_argv[] = {"git", "-C", repo, "commit", "-q", "-m", "merge source", NULL};
    run_cmd(add_merge_argv);
    run_cmd(commit_merge_argv);
    char *switch_side_argv[] = {"git", "-C", repo, "switch", "-q", "side", NULL};
    run_cmd(switch_side_argv);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "merge_base",
        .base_ref = "side",
        .target_ref = "merge_src",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strlen(r.output) >= 40, "merge_base did not return an oid");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "merge_preview",
        .target_ref = "merge_src",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strlen(r.output) >= 40, "merge_preview did not return a merge tree");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "merge",
        .target_ref = "merge_src",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "merge should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "merge",
        .target_ref = "merge_src",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strlen(r.output) >= 40, "merge dry-run did not return a merge tree");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "merge",
        .target_ref = "merge_src",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "Fast-forward") != NULL, "merge did not fast-forward");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "file_at_ref",
        .ref = "HEAD",
        .path = "d.txt",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(!strcmp(r.output, "merge\n"), "merge did not bring target file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "merge_abort",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    char *topic_branch_argv[] = {"git", "-C", repo, "switch", "-q", "-c", "topic", NULL};
    run_cmd(topic_branch_argv);
    write_file(topic_file, "topic\n");
    char *add_topic_argv[] = {"git", "-C", repo, "add", "e.txt", NULL};
    char *commit_topic_argv[] = {"git", "-C", repo, "commit", "-q", "-m", "topic commit", NULL};
    run_cmd(add_topic_argv);
    run_cmd(commit_topic_argv);
    run_cmd(switch_side_argv);
    write_file(base_file, "base\n");
    char *add_base_argv[] = {"git", "-C", repo, "add", "f.txt", NULL};
    char *commit_base_argv[] = {"git", "-C", repo, "commit", "-q", "-m", "base advance", NULL};
    run_cmd(add_base_argv);
    run_cmd(commit_base_argv);
    char *switch_topic_argv[] = {"git", "-C", repo, "switch", "-q", "topic", NULL};
    run_cmd(switch_topic_argv);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "rebase_preview",
        .ref = "side",
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "topic commit") != NULL, "rebase_preview missing topic commit");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "rebase", .ref = "side"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "rebase should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "rebase",
        .ref = "side",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "topic commit") != NULL, "rebase dry-run missing topic commit");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "rebase",
        .ref = "side",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "file_at_ref", .ref = "HEAD", .path = "e.txt"};
    assert_git_opts_ok(&opts, &r);
    CHECK(!strcmp(r.output, "topic\n"), "rebase lost topic file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "file_at_ref", .ref = "HEAD", .path = "f.txt"};
    assert_git_opts_ok(&opts, &r);
    CHECK(!strcmp(r.output, "base\n"), "rebase did not include upstream file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "rebase_abort",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    write_file(file, "stashed\n");
    write_file(stash_file, "extra\n");
    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_push",
        .message = "save local state",
        .all = true,
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "M a.txt") != NULL, "stash dry-run missing tracked file");
    CHECK(strstr(r.output, "?? c.txt") != NULL, "stash dry-run missing untracked file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_push",
        .message = "save local state",
        .all = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "save local state") != NULL, "stash_push missing stash message");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "changed_files"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "a.txt") == NULL, "stash_push left tracked file dirty");
    CHECK(strstr(r.output, "c.txt") == NULL, "stash_push left untracked file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "stash_list", .limit = 5};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "save local state") != NULL, "stash_list missing saved state");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_show",
        .ref = "stash@{0}",
        .patch = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "+stashed") != NULL, "stash_show patch missing tracked change");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_apply",
        .ref = "stash@{0}",
        .dry_run = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "a.txt") != NULL, "stash_apply dry-run missing staged summary");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_apply",
        .ref = "stash@{0}",
    };
    assert_git_opts_ok(&opts, &r);
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){.repo = repo, .action = "changed_files"};
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, " M a.txt") != NULL, "stash_apply did not restore tracked change");
    CHECK(strstr(r.output, "?? c.txt") != NULL, "stash_apply did not restore untracked file");
    ds4_agent_git_result_free(&r);

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_drop",
        .ref = "stash@{0}",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "stash_drop should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_pop",
        .ref = "stash@{0}",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "stash_pop should require confirm=true or dry_run=true");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_drop",
        .ref = "stash@{0}",
        .confirm = true,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "Dropped") != NULL, "stash_drop did not drop stash");
    ds4_agent_git_result_free(&r);

    CHECK(!ds4_agent_git_run(repo, "show", NULL, "--stat", 0, false,
                             64 * 1024, &r, err, sizeof(err)),
          "unsafe ref should be rejected before git runs");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "file_at_ref",
        .ref = "HEAD:a.txt",
        .path = "a.txt",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "file_at_ref should reject refs containing a tree separator");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "blame",
        .ref = "--reverse",
        .path = "a.txt",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "blame should reject unsafe refs");

    opts = (ds4_agent_git_options){.repo = repo, .action = "stage"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "stage should require path or all=true");

    opts = (ds4_agent_git_options){.repo = repo, .action = "unstage"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "unstage should require path or all=true");

    opts = (ds4_agent_git_options){.repo = repo, .action = "commit"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "commit should require message");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "commit",
        .message = "bad\nmessage",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "commit should reject multiline messages");

    opts = (ds4_agent_git_options){.repo = repo, .action = "worktree_restore"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "worktree_restore should require path or all=true");

    opts = (ds4_agent_git_options){.repo = repo, .action = "switch"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "switch should require ref");

    opts = (ds4_agent_git_options){.repo = repo, .action = "switch", .ref = "--detach"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "switch should reject unsafe refs");

    opts = (ds4_agent_git_options){.repo = repo, .action = "stash_push"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "stash_push should require message");

    opts = (ds4_agent_git_options){.repo = repo, .action = "merge"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "merge should require target ref");

    opts = (ds4_agent_git_options){.repo = repo, .action = "merge", .target_ref = "--strategy"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "merge should reject unsafe target refs");

    opts = (ds4_agent_git_options){.repo = repo, .action = "rebase"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "rebase should require upstream ref");

    opts = (ds4_agent_git_options){.repo = repo, .action = "rebase", .ref = "--onto"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "rebase should reject unsafe upstream refs");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "stash_show",
        .ref = "HEAD",
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "stash_show should reject non-stash refs");

    opts = (ds4_agent_git_options){.repo = repo, .action = "fetch"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "fetch should require remote");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "fetch",
        .remote = "--upload-pack",
        .dry_run = true,
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "fetch should reject unsafe remotes");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "push",
        .remote = "local",
        .ref = ":side",
        .dry_run = true,
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "push should reject delete refspecs");

    opts = (ds4_agent_git_options){.repo = repo, .action = "diff", .range = "--stat"};
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "unsafe range should be rejected before git runs");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "diff",
        .name_only = true,
        .name_status = true,
    };
    CHECK(!ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "conflicting diff formats should be rejected");

    char *fake_dir = make_temp_dir();
    char *fake_git = join_path(fake_dir, "git");
    const char *current_path = getenv("PATH");
    bool had_path = current_path != NULL;
    char *old_path = current_path ? test_strdup(current_path) : NULL;
    size_t new_path_len = strlen(fake_dir) + 1 + (old_path ? strlen(old_path) : 0) + 1;
    char *new_path = malloc(new_path_len);
    CHECK(new_path != NULL, "malloc failed");
    if (old_path && old_path[0])
        snprintf(new_path, new_path_len, "%s:%s", fake_dir, old_path);
    else
        snprintf(new_path, new_path_len, "%s", fake_dir);

    write_file(fake_git,
               "#!/bin/sh\n"
               "printf 'prompt=%s askpass=%s ssh=%s editor=%s autoedit=%s gcm=%s\\n' "
               "\"$GIT_TERMINAL_PROMPT\" \"$GIT_ASKPASS\" \"$SSH_ASKPASS\" "
               "\"$GIT_EDITOR\" \"$GIT_MERGE_AUTOEDIT\" \"$GCM_INTERACTIVE\"\n");
    CHECK(chmod(fake_git, 0700) == 0, "failed to chmod fake git");
    CHECK(setenv("PATH", new_path, 1) == 0, "failed to override PATH");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "status",
        .timeout_sec = 5,
    };
    assert_git_opts_ok(&opts, &r);
    CHECK(strstr(r.output, "prompt=0") != NULL, "git should disable terminal prompts");
    CHECK(strstr(r.output, "askpass=/bin/false") != NULL, "git should disable askpass");
    CHECK(strstr(r.output, "ssh=/bin/false") != NULL, "git should disable ssh askpass");
    CHECK(strstr(r.output, "editor=true") != NULL, "git should use a noninteractive editor");
    CHECK(strstr(r.output, "autoedit=no") != NULL, "git should disable merge autoedit");
    CHECK(strstr(r.output, "gcm=never") != NULL, "git should disable credential manager prompts");
    ds4_agent_git_result_free(&r);

    write_file(fake_git,
               "#!/bin/sh\n"
               "sleep 2\n"
               "echo late\n");
    CHECK(chmod(fake_git, 0700) == 0, "failed to chmod timeout fake git");

    opts = (ds4_agent_git_options){
        .repo = repo,
        .action = "status",
        .timeout_sec = 1,
    };
    err[0] = '\0';
    CHECK(ds4_agent_git_run_options(&opts, &r, err, sizeof(err)),
          "timeout fake git should execute");
    CHECK(r.exit_code == 124, "timed out git command should report exit code 124");
    CHECK(strstr(r.output, "timed out after 1 seconds") != NULL,
          "timed out git command should report timeout");
    CHECK(strstr(r.output, "late") == NULL,
          "timed out git command should kill the process group");
    ds4_agent_git_result_free(&r);

    if (had_path)
        CHECK(setenv("PATH", old_path, 1) == 0, "failed to restore PATH");
    else
        CHECK(unsetenv("PATH") == 0, "failed to unset PATH");
    remove_tree(fake_dir);
    free(new_path);
    free(old_path);
    free(fake_git);
    free(fake_dir);

    remove_tree(repo);
    remove_tree(remote_dir);
    free(base_file);
    free(topic_file);
    free(merge_file);
    free(stash_file);
    free(untracked);
    free(file);
    free(remote_dir);
    free(repo);
    return 0;
}
