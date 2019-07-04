If use TB-RK1808 AI Compute Stick or RK1808(with NTB mode) in Linux OS, such as Ubuntu, we need execute command 'sduo update_rk1808_usb_rule.sh' to get read/write permission of device.

For example:
rk@rk:~/rknn-toolkit-v1.1.0/platform-tools/update_rk_usb_rule/linux$ sudo update_rk1808_usb_rule.sh
Note: this command just need excute one time when we install RKNN-Toolkit.


After execute this command, we can check the read/write permisiion of RK1808 or TB-RK1808 AI Compute Stick like below:
rk@rk:~$ ll /dev/bus/usb/003/007
crw-rw-rw- 1 root root 189, 330 May 15 09:37 /dev/bus/usb/003/075
Note: 003 is bus number, 075 is device number, we can get these numbers through executing 'lsusb' command.
