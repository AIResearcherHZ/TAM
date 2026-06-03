# TAM Deployment

This repo includes the workstation-side TAM mapping server and the external
trajectory-evaluation wrapper. The low-level robot controller bridge runs on the
deployment controller machine and is not included here.

## Connection Roles

| Endpoint | Default | Direction | Purpose |
| --- | --- | --- | --- |
| `--history-endpoint` | `tcp://192.168.1.101:5555` | controller PUB -> mapping/eval SUB | Streams history windows, reset events, joint state, and torque fields. |
| `--command-endpoint` | `tcp://192.168.1.101:5556` | mapping/eval PUSH -> controller PULL | Sends target commands, TAM embeddings, and feedforward fields. |
| `--request-endpoint` | `tcp://192.168.1.101:5557` | mapping/eval REQ -> controller REP | Sends reliable requests such as loading TAM weights and enabling/disabling TAM. |
| `--control-endpoint` | `tcp://0.0.0.0:5560` | operator/eval REQ -> mapping server REP | Sends direct mapping-server commands such as reset, hold, resume, and status. |

Use controller-host addresses for the first three endpoints. Use a bind address
for `--control-endpoint` on the workstation that runs `tam-mapping-server`.

## Mapping Server

```bash
tam-mapping-server \
  --ckpt-path checkpoints/tam/tam_rby1 \
  --xml assets/rby1a/rby1_onearm.xml \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557 \
  --control-endpoint tcp://0.0.0.0:5560 \
  --require-control-enable
```

The server waits for controller history, prepares the TAM runtime, uploads the
exported TAM weight blob when `--send-bin` is enabled, and only enables TAM after
the first valid embedding unless configured otherwise.

`--require-control-enable` keeps TAM disabled until an operator or evaluation
wrapper sends a control command to the mapping-server control endpoint. This is
the recommended deployment default because it prevents stale history from
enabling TAM before the robot is homed.

## Evaluation Wrapper

```bash
tam-eval-wrapper \
  --reference sine \
  --history-endpoint tcp://<controller-host>:5555 \
  --command-endpoint tcp://<controller-host>:5556 \
  --request-endpoint tcp://<controller-host>:5557
```

The wrapper talks to the same controller bridge as the mapping server. Keep the
mapping server running when evaluating TAM-enabled control; run the wrapper
without a TAM mapping server only for direct-controller comparisons.

## Startup Order

1. Start the deployment-side controller bridge on the robot controller machine.
2. Start `tam-mapping-server` on the workstation with the matching checkpoint
   and robot XML.
3. Confirm the mapping server reports fresh history and a prepared TAM runtime.
4. Send a reset/resume command through the mapping-server control endpoint or
   start `tam-eval-wrapper`.
5. Stop the evaluation wrapper before stopping the mapping server or controller
   bridge.

## Network Notes

- Keep clocks reasonably synchronized so history timestamps and evaluation logs
  are easy to inspect.
- The controller bridge should bind the history, command, and request ports.
  The workstation connects to those ports.
- The mapping server binds the control port. Operator tools and evaluation
  scripts connect to it.
- Use SSH tunnels or firewall rules when the controller network is not directly
  reachable from the workstation.
