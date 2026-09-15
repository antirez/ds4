# GNU make prefers this file over Makefile. Keep all upstream targets and only
# replace the two ds4-agent compilation rules with the hook-enabled translation
# unit. Other binaries and backends remain unchanged.
include Makefile

DS4_AGENT_HOOK_SRCS := ds4_agent_with_hooks.c ds4_agent.c ds4_agent_hooks.c ds4_hooks.c ds4_hooks.h

ds4_agent.o: $(DS4_AGENT_HOOK_SRCS) ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent_with_hooks.c

ds4_agent_cpu.o: $(DS4_AGENT_HOOK_SRCS) ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent_with_hooks.c

.PHONY: test-hooks

tests/ds4_hooks_test: tests/ds4_hooks_test.c ds4_hooks.c ds4_hooks.h
	$(CC) $(CFLAGS) -I. -o $@ tests/ds4_hooks_test.c ds4_hooks.c $(LDLIBS)

test-hooks: tests/ds4_hooks_test
	./tests/ds4_hooks_test
