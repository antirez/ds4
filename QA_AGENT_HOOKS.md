# Agent hooks QA

- [ ] `make test-hooks`
- [ ] `make cpu`
- [ ] `./ds4-agent --help hooks` lists all three options
- [ ] no hook command means no child process is started
- [ ] before event contains user text and no response text
- [ ] after event contains reconstructed response text and token count
- [ ] non-zero exits are reported without ending the session
- [ ] timeout kills the command process group
- [ ] interrupted responses set `interrupted: true`
- [ ] tool turns set `tool_call: true`
- [ ] quotes, backslashes, control bytes, and UTF-8 serialize correctly
