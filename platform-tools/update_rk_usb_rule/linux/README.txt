如果在Ubuntu等Linux系统上使用TB-RK1808 AI计算棒或RK1808(NTB模式),需要以sudo权限执行update_rk1808_usb_rule.sh脚本以获取USB设备的读写权限.
示例如下:
rk@rk:~/rknn-toolkit-v1.1.0/platform-tools/update_rk_usb_rule/linux$ sudo update_rk1808_usb_rule.sh

注: 这一步只需要在安装RKNN-Toolkit后执行一次即可,以后不需要再执行.

执行完脚本后,我们查看RK1808或TB-RK1808 AI计算棒设备的读写权限,应该如下所示:
rk@rk:~$ ll /dev/bus/usb/003/075
crw-rw-rw- 1 root root 189, 330 May 15 09:37 /dev/bus/usb/003/075

其中003是该设备的bus号,075是device号,这些数字可以通过lsusb查到.
