import serial
import threading
import time
import sys
from datetime import datetime


class STM32ScissorController:
    def __init__(self, port='COM3', baudrate=115200):
        """
        初始化STM32剪刀控制器
        :param port: 串口端口
        :param baudrate: 波特率，默认为115200
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_port = None
        self.running = True
        self.scissor_active = False  # 剪刀系统是否激活
        self.auto_mode = False  # 自动发送模式
        self.auto_interval = 2.0  # 自动发送间隔(秒)
        self.auto_thread = None
        self.data_buffer = ""
        self.input_available = False  # 用于标记是否有用户输入

        # 新增：休眠状态管理
        self.last_command_time = time.time()  # 最后指令时间
        self.idle_timeout = 60  # 休眠超时时间（秒）
        self.sleep_mode = False  # 是否处于休眠状态

    def connect(self):
        """连接串口"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )

            if self.serial_port.is_open:
                print(f"成功连接到串口 {self.port}，波特率 {self.baudrate}")
                # 等待STM32初始化完成
                time.sleep(2)
                # 清空缓冲区
                self.serial_port.flushInput()
                self.serial_port.flushOutput()
                self.last_command_time = time.time()  # 重置最后指令时间
                return True

        except serial.SerialException as e:
            print(f"无法打开串口 {self.port}: {e}")
            return False
        except Exception as e:
            print(f"连接错误: {e}")
            return False

    def disconnect(self):
        """断开串口连接"""
        self.stop_auto_mode()  # 先停止自动模式
        if self.serial_port and self.serial_port.is_open:
            # 发送停止命令确保设备安全
            self.send_command("stop")
            time.sleep(0.1)
            self.serial_port.close()
            print("串口连接已关闭")

    def update_command_time(self):
        """更新最后指令时间"""
        self.last_command_time = time.time()

    def check_sleep_status(self):
        """检查是否进入休眠状态"""
        current_time = time.time()
        idle_time = current_time - self.last_command_time

        if idle_time >= self.idle_timeout and not self.sleep_mode:
            print(f"设备空闲超过{self.idle_timeout}秒，进入休眠状态")
            self.sleep_mode = True
            self.scissor_active = False
        elif idle_time < self.idle_timeout and self.sleep_mode:
            self.sleep_mode = False

        return self.sleep_mode

    def send_command(self, command):
        """发送命令到STM32"""
        if self.sleep_mode:
            print("设备处于休眠状态，请先唤醒设备")
            return False

        if self.serial_port and self.serial_port.is_open:
            try:
                # 添加换行符作为命令结束符
                cmd = command + '\n'
                self.serial_port.write(cmd.encode('utf-8'))
                self.serial_port.flush()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送命令: {command}")

                # 更新最后指令时间
                self.update_command_time()
                return True
            except Exception as e:
                print(f"发送命令失败: {e}")
                return False
        else:
            print("串口未连接")
            return False

    def receive_data(self):
        """接收STM32发送的数据"""
        if self.serial_port and self.serial_port.is_open:
            try:
                # 读取所有可用数据
                while self.serial_port.in_waiting > 0:
                    data = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if data:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 接收: {data}")
                        self.data_buffer = data
                return self.data_buffer
            except Exception as e:
                print(f"接收数据失败: {e}")
                return None

    def start_receiver_thread(self):
        """启动接收线程"""

        def receiver():
            while self.running:
                self.receive_data()
                time.sleep(0.01)

        receiver_thread = threading.Thread(target=receiver, daemon=True)
        receiver_thread.start()
        print("接收线程已启动")

    def power_on(self):
        """发送开机命令"""
        print("发送开机命令: ON")
        success = self.send_command("ON")
        if success:
            self.scissor_active = True
            self.sleep_mode = False
        return success

    def power_off(self):
        """发送关机命令"""
        print("发送关机命令: OFF")
        success = self.send_command("OFF")
        if success:
            self.scissor_active = False
            self.sleep_mode = True
            self.stop_auto_mode()  # 关机时停止自动模式
        return success

    def activate_scissor_once(self):
        """激活剪刀一次"""
        if self.scissor_active:
            print("发送剪刀激活命令: 0")
            success = self.send_command("0")
            return success
        else:
            print("错误: 剪刀系统未激活")
            return False

    def stop_control(self):
        """发送停止命令"""
        print("发送停止命令: stop")
        success = self.send_command("stop")
        if success:
            self.scissor_active = False
        self.stop_auto_mode()
        return success

    def start_auto_mode(self, interval=2.0):
        """启动自动发送模式"""
        if not self.scissor_active:
            print("错误: 请先发送ON开机，才能启动自动模式")
            return False

        if self.auto_mode:
            print("自动模式已经在运行")
            return True

        print(f"启动自动模式，每隔 {interval} 秒发送一次 '0'")

        self.auto_mode = True
        self.auto_interval = interval

        def auto_sender():
            counter = 0
            while self.auto_mode and self.running and self.scissor_active:
                try:
                    counter += 1
                    print(f"[自动模式] 第 {counter} 次发送 '0'")
                    self.send_command("0")

                    # 等待指定的间隔时间
                    wait_time = 0
                    while wait_time < self.auto_interval and self.auto_mode:
                        time.sleep(0.1)
                        wait_time += 0.1

                except Exception as e:
                    print(f"自动发送出错: {e}")
                    break

            print("自动模式已停止")

        self.auto_thread = threading.Thread(target=auto_sender, daemon=True)
        self.auto_thread.start()
        return True

    def stop_auto_mode(self):
        """停止自动发送模式"""
        if self.auto_mode:
            print("停止自动模式...")
            self.auto_mode = False
            if self.auto_thread:
                self.auto_thread.join(timeout=1.0)
        return True

    # ==================== 新增接口函数 ====================

    def checkstate(self):
        """
        检查工作状态
        返回1代表工作，返回0代表不正常
        一分钟内没有任何指令下发默认进入休眠状态
        """
        # 检查休眠状态
        is_sleeping = self.check_sleep_status()

        if is_sleeping:
            print("设备处于休眠状态")
            return 0
        elif self.scissor_active:
            print("设备处于工作状态")
            return 1
        else:
            print("设备未激活")
            return 0

    def releaseshear(self):
        """
        打开一次剪刀
        成功启动返回1，不成功返回0
        """
        # 检查是否处于休眠状态
        if self.check_sleep_status():
            print("剪刀处于休眠状态，请先唤醒")
            return 0

        # 检查是否已激活
        if not self.scissor_active:
            print("剪刀未激活，请先唤醒")
            return 0

        # 发送剪刀激活命令
        success = self.activate_scissor_once()

        if success:
            print("剪刀成功激活一次")
            return 1
        else:
            print("剪刀激活失败")
            return 0

    def shearawake(self):
        """
        唤醒剪刀
        返回1代表唤醒成功，0代表失败
        """
        # 如果已经处于工作状态，直接返回成功
        if self.scissor_active and not self.sleep_mode:
            print("剪刀已处于唤醒状态")
            return 1

        # 发送ON命令唤醒设备
        success = self.power_on()

        if success:
            print("剪刀唤醒成功")
            return 1
        else:
            print("剪刀唤醒失败")
            return 0

    def shearshutdown(self):
        """
        关闭剪刀（关机）
        返回1代表关机成功，0代表失败
        """
        # 如果已经处于关闭状态，直接返回成功
        if not self.scissor_active or self.sleep_mode:
            print("剪刀已处于关闭状态")
            return 1

        # 发送OFF命令关闭设备
        success = self.power_off()

        if success:
            print("剪刀关闭成功")
            return 1
        else:
            print("剪刀关闭失败")
            return 0

    # ==================== 原有功能保持不变 ====================

    def interactive_control(self):
        """交互式控制界面 - 模式1：保持原有简单交互"""
        print("\n" + "=" * 50)
        print("STM32剪刀控制系统 - 交互模式")
        print("=" * 50)
        print("命令说明:")
        print("  'ON'   - 开机")
        print("  '0'    - 开启一次剪刀")
        print("  'stop' - 停止控制")
        print("  'quit' - 退出程序")
        print("  'help' - 显示帮助")
        print("=" * 50)
        print("注意: 必须先发送'ON'开机，然后才能发送'0'激活剪刀")
        print("=" * 50)

        # 启动数据接收线程
        self.start_receiver_thread()

        while self.running:
            try:
                # 获取用户输入
                user_input = input("\n请输入命令: ").strip()

                if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                    print("正在退出程序...")
                    self.running = False
                    self.stop_control()
                    break

                elif user_input.lower() == 'help':
                    print("\n命令说明:")
                    print("  ON   - 开机（必须首先执行）")
                    print("  0    - 开启一次剪刀（开机后才能使用）")
                    print("  stop - 停止控制")
                    print("  quit - 退出程序")
                    print("\n使用流程:")
                    print("  1. 首先发送 'ON' 开机")
                    print("  2. 然后可以多次发送 '0' 激活剪刀")
                    print("  3. 发送 'stop' 停止控制")
                    print("  4. 发送 'quit' 退出程序")

                elif user_input == 'ON':
                    self.power_on()

                elif user_input == '0':
                    self.activate_scissor_once()

                elif user_input == 'stop':
                    self.stop_control()

                elif user_input == '':
                    continue  # 忽略空输入

                else:
                    print(f"未知命令: {user_input}")
                    print("请输入 'ON', '0', 'stop' 或 'quit'")
                    print("输入 'help' 查看帮助")

                # 短暂延迟
                time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n检测到Ctrl+C，正在退出...")
                self.running = False
                self.stop_control()
                break
            except Exception as e:
                print(f"命令执行错误: {e}")

    def auto_scissor_demo(self):
        """自动演示模式 - 模式2"""
        print("\n启动自动剪刀演示模式...")
        print("将按以下顺序执行:")
        print("1. 发送ON开机")
        print("2. 等待2秒")
        print("3. 连续激活剪刀5次")
        print("4. 发送stop停止")

        # 启动数据接收线程
        self.start_receiver_thread()
        time.sleep(1)

        try:
            # 1. 发送开机命令
            print("\n[步骤1] 发送开机命令")
            self.power_on()
            time.sleep(2)

            # 2. 连续激活剪刀
            print("\n[步骤2] 开始连续激活剪刀")
            for i in range(5):
                if not self.running:
                    break

                print(f"\n剪刀激活 #{i + 1}")
                self.activate_scissor_once()
                time.sleep(3)  # 每次激活间隔3秒

            # 3. 发送停止命令
            print("\n[步骤3] 发送停止命令")
            self.stop_control()
            time.sleep(1)

            print("\n自动演示完成！")

        except KeyboardInterrupt:
            print("\n自动模式被中断")
            self.stop_control()

    def continuous_auto_mode(self):
        """模式3：ON启动后，发送第一个0，然后自动持续发送"""
        print("\n" + "=" * 60)
        print("模式3：持续自动模式")
        print("=" * 60)
        print("模式说明:")
        print("1. 首先发送ON开机")
        print("2. 手动发送第一个'0'")
        print("3. 自动启动连续发送模式")
        print("4. 每隔2秒自动发送'0'")
        print("5. 输入'stop'停止自动模式")
        print("6. 输入'quit'退出程序")
        print("=" * 60)

        # 启动数据接收线程
        self.start_receiver_thread()

        # 第一步：发送ON开机
        print("\n[第一步] 请发送ON开机")
        while True:
            cmd = input("请输入命令 (输入 'ON' 或 'quit'): ").strip()

            if cmd.lower() == 'quit':
                print("退出程序")
                self.running = False
                return
            elif cmd == 'ON':
                self.power_on()
                break
            else:
                print("请先发送ON开机")

        # 等待系统就绪
        time.sleep(1)

        # 第二步：手动发送第一个0
        print("\n[第二步] 请发送第一个'0'")
        while True:
            cmd = input("请输入命令 (输入 '0' 或 'quit'): ").strip()

            if cmd.lower() == 'quit':
                self.stop_control()
                self.running = False
                return
            elif cmd == '0':
                if self.activate_scissor_once():
                    break
            else:
                print("请输入'0'或'quit'")

        # 第三步：询问自动模式参数
        print("\n[第三步] 设置自动模式参数")
        interval = 2.0
        while True:
            try:
                interval_input = input(f"请输入自动发送间隔秒数 (默认{interval}秒，直接回车使用默认): ").strip()
                if interval_input == "":
                    break
                interval = float(interval_input)
                if 0.1 <= interval <= 10.0:
                    break
                else:
                    print("间隔应在0.1-10.0秒之间")
            except ValueError:
                print("请输入有效的数字")

        # 第四步：启动自动模式
        print(f"\n[第四步] 启动自动模式，每隔 {interval} 秒发送一次'0'")
        print("自动模式运行中...")
        print("按Enter键停止自动模式并返回控制")

        if not self.start_auto_mode(interval=interval):
            print("启动自动模式失败")
            return

        # 第五步：简化版输入检查 - 等待用户按Enter
        try:
            # 简单方法：等待用户按Enter键
            input()  # 这会阻塞，直到用户按Enter

            print("\n检测到用户输入，停止自动模式...")
            self.stop_auto_mode()

        except KeyboardInterrupt:
            print("\n检测到Ctrl+C")
            self.running = False
            self.stop_control()
            return

        # 自动模式停止后的处理
        if self.running:
            print("\n自动模式已停止")
            print("现在可以继续手动控制:")
            print("1. 输入 '0' 手动激活剪刀")
            print("2. 输入 'stop' 停止系统")
            print("3. 输入 'quit' 退出程序")

            while self.running:
                try:
                    cmd = input("\n请输入命令: ").strip()

                    if cmd.lower() == 'quit':
                        self.stop_control()
                        self.running = False
                        break
                    elif cmd == '0':
                        self.activate_scissor_once()
                    elif cmd == 'stop':
                        self.stop_control()
                        print("系统已停止")
                        # 是否继续？
                        restart = input("是否重新开始? (y/n): ").lower()
                        if restart == 'y':
                            # 重新开始模式3
                            self.continuous_auto_mode()
                            return
                        else:
                            break
                    else:
                        print("未知命令，请输入 '0', 'stop', 或 'quit'")

                except KeyboardInterrupt:
                    print("\n检测到Ctrl+C")
                    self.stop_control()
                    self.running = False
                    break

    def cleanup(self):
        """清理资源"""
        self.running = False
        self.stop_auto_mode()
        self.disconnect()
        print("程序已安全退出")


def list_serial_ports():
    """列出可用的串口"""
    try:
        from .main import list_serial_ports as _list_serial_ports
    except ImportError:
        from main import list_serial_ports as _list_serial_ports
    return _list_serial_ports()


def main():
    """主函数"""
    print("=" * 50)
    print("STM32剪刀串口控制系统")
    print("=" * 50)

    # 列出可用串口
    print("\n扫描可用串口...")
    available_ports = list_serial_ports()

    if not available_ports:
        print("未找到可用串口！")
        print("请检查:")
        print("1. STM32是否已连接到电脑")
        print("2. USB转串口驱动是否安装")
        print("3. 串口线是否正常")
        return

    print("找到以下串口:")
    for i, port in enumerate(available_ports):
        print(f"  [{i}] {port}")

    # 选择串口
    try:
        if len(available_ports) == 1:
            selected_port = available_ports[0]
            print(f"\n自动选择串口: {selected_port}")
        else:
            choice = int(input(f"\n请选择串口编号 (0-{len(available_ports) - 1}): "))
            selected_port = available_ports[choice]
    except (ValueError, IndexError):
        print("选择无效，使用列表中的第一个串口")
        selected_port = available_ports[0]

    # 创建控制器并连接
    controller = STM32ScissorController(port=selected_port, baudrate=115200)

    if not controller.connect():
        print("连接失败，程序退出")
        return

    try:
        # 选择模式
        print("\n请选择控制模式:")
        print("1. 交互模式 (手动输入所有命令)")
        print("2. 自动演示模式 (观看完整演示)")
        print("3. 持续自动模式 (ON→第一个0→自动连续发送0)")
        print("4. 测试新接口")

        try:
            mode_choice = int(input("请选择模式 (1/2/3/4): "))
        except ValueError:
            mode_choice = 1

        if mode_choice == 2:
            # 自动演示模式
            controller.auto_scissor_demo()
        elif mode_choice == 3:
            # 持续自动模式
            controller.continuous_auto_mode()
        elif mode_choice == 4:
            # 测试新接口
            test_new_interface(controller)
        else:
            # 交互模式（保持不变）
            controller.interactive_control()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 确保清理资源
        controller.cleanup()


def test_new_interface(controller):
    """测试新接口函数"""
    print("\n" + "=" * 60)
    print("测试新接口功能")
    print("=" * 60)
    print("将测试以下封装函数:")
    print("1. checkstate()    - 检查工作状态")
    print("2. shearawake()    - 唤醒剪刀（开机）")
    print("3. releaseshear()  - 打开一次剪刀")
    print("4. shearshutdown() - 关闭剪刀（关机）")
    print("5. 返回主菜单")
    print("=" * 60)

    while True:
        print("\n选择要测试的功能:")
        print("1. checkstate() - 检查工作状态")
        print("2. shearawake() - 唤醒剪刀（开机）")
        print("3. releaseshear() - 打开一次剪刀")
        print("4. shearshutdown() - 关闭剪刀（关机）")
        print("5. 完整流程测试")
        print("6. 返回主菜单")

        try:
            choice = input("请选择 (1-6): ").strip()
        except KeyboardInterrupt:
            print("\n返回主菜单")
            break

        if choice == '1':
            print("\n[测试] checkstate() - 检查工作状态:")
            state = controller.checkstate()
            print(f"返回值: {state} (1=工作, 0=休眠/关闭)")
            if state == 1:
                print("当前状态: 设备已开机")
            else:
                print("当前状态: 设备已关机或休眠")

        elif choice == '2':
            print("\n[测试] shearawake() - 唤醒剪刀（开机）:")
            result = controller.shearawake()
            print(f"返回值: {result} (1=成功, 0=失败)")
            if result == 1:
                print("开机成功")
            else:
                print("开机失败")

        elif choice == '3':
            print("\n[测试] releaseshear() - 打开一次剪刀:")
            # 先检查状态
            state = controller.checkstate()
            if state == 0:
                print("设备未开机，请先开机")
                continue

            result = controller.releaseshear()
            print(f"返回值: {result} (1=成功, 0=失败)")
            if result == 1:
                print("剪刀动作成功")
            else:
                print("剪刀动作失败")

        elif choice == '4':
            print("\n[测试] shearshutdown() - 关闭剪刀（关机）:")
            result = controller.shearshutdown()
            print(f"返回值: {result} (1=成功, 0=失败)")
            if result == 1:
                print("关机成功")
            else:
                print("关机失败")

        elif choice == '5':
            print("\n[测试] 完整流程测试:")
            print("1. 检查初始状态...")
            state1 = controller.checkstate()
            print(f"   初始状态: {state1}")

            print("2. 开机...")
            if state1 == 0:
                wake_result = controller.shearawake()
                print(f"   开机结果: {wake_result}")
            else:
                print("   设备已开机")

            print("3. 检查开机后状态...")
            state2 = controller.checkstate()
            print(f"   开机后状态: {state2}")

            if state2 == 1:
                print("4. 剪刀动作测试...")
                for i in range(3):
                    print(f"   第{i + 1}次剪刀动作:")
                    cut_result = controller.releaseshear()
                    print(f"     结果: {cut_result}")
                    if i < 2:
                        time.sleep(1)

            print("5. 关机...")
            shutdown_result = controller.shearshutdown()
            print(f"   关机结果: {shutdown_result}")

            print("6. 最终状态检查...")
            final_state = controller.checkstate()
            print(f"   最终状态: {final_state}")

            print("\n完整流程测试完成!")

        elif choice == '6':
            print("返回主菜单")
            break

        else:
            print("无效选择，请输入1-6")

        # 等待一下
        time.sleep(1)


if __name__ == "__main__":
    # 检查是否安装pyserial
    try:
        import serial
    except ImportError:
        print("错误: 未找到pyserial库")
        print("请先安装: pip install pyserial")
        sys.exit(1)

    main()
