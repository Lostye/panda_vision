#ifndef SERIAL
#define SERIAL
#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<errno.h>
#include<time.h>
#include<string.h>
#include<termios.h>
#include<iostream>
#include<unistd.h>

using namespace std;

class serial{
private:
    typedef struct {
        unsigned char data_all[10] = { 0xED,0x00,0x00,0x00,0x00,/*yaw*/0x00,0x00,0x00,0x00,/*pitch*/0xEC };
    }send_data;//发送的数据

    union {
        int I_Point;//浮点型yaw
        u_char U_Point[4];//无字符型yaw
    }SendPoint;



    union {
        int I_mode;//浮点型yaw
        u_char U_mode[4];//无字符型yaw
    }Sendmode;






public:

    float getYaw=0,getPitch=0,getSpeed=10/*,getYaw_angle=0,getPitch_angle=0*/;
    bool getColor;//red 1  bule 0;
    int getState=0;
    bool If_Buff=0;//是否收到开启风车模式的指令
    u_char Getpanda=0;

    bool port_init(int &port_fd);
    int port_SEND(int &fd, int point , int mod);
    bool port_RECV(int &fd);
    //下面是实现功能的简单模块
private:
    int OPEN(  char *port);//打开串口模块
    bool SET(int fd);//设置模块
    int SEND(int fd, send_data data);//发送模块
};
//bool port_RECV_RUN();
#endif


