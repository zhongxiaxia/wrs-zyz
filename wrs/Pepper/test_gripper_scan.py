# save as test_gripper_scan.py
from jodellSdk.jodellSdkDemo import ClawHepgTool

PORTS = [
    "/dev/ttyTHS1", "/dev/ttyTHS2",
    "/dev/ttyS0", "/dev/ttyS1", "/dev/ttyS2", "/dev/ttyS3",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6",
]
BAUDS = [115200, 57600, 9600]
IDS = range(1, 17)

tool = ClawHepgTool()

for port in PORTS:
    for baud in BAUDS:
        print(f"\n=== Try port={port}, baud={baud} ===")
        try:
            ret = tool.serialOperation(port, baud, True)
            print(f"serialOperation -> {ret}")
            if ret != 1:
                continue

            for claw_id in IDS:
                try:
                    r = tool.clawEnable(claw_id, True)
                    print(f"  id={claw_id} enable -> {r}")
                    if r == 1:
                        print(f"\n[FOUND] port={port}, baud={baud}, claw_id={claw_id}")
                        tool.serialOperation(port, baud, False)
                        raise SystemExit(0)
                except Exception as e:
                    print(f"  id={claw_id} exception: {e}")

            tool.serialOperation(port, baud, False)
        except Exception as e:
            print(f"open exception: {e}")

print("\n[NOT FOUND] 没扫到可通信参数（大概率是接线/RS485转换/奇偶校验问题）")
