#include "serial.h"

bool serial::port_init(int &fd)
{
    int is_set=0;
    char data_name[] = { "/dev/ttyUSB0" };
    if(fd<=0){
        fd =OPEN(data_name);
    }
    if (fd == -1) return false;
    while (!is_set) {
        close(fd);
        fd = OPEN(data_name);
        is_set=SET(fd);
        if(is_set)
        printf("port set success\n");
    }
    return false;
}


/*
bool port_RECV_RUN(){
    int is_set=0;
    char data_name[] = { "/dev/ttyUSB0" };
    int fd = port_OPEN(data_name);
    if (fd == -1) return false;
    while (!is_set) {
        is_set=port_SET(fd);
        printf("port set success\n");
    }
    char receve[20];
    if(!port_RECV(fd,receve)){
        cout<<"recv false"<<endl;
        return false;
    }
     getYaw=float((256*int8_t(receve[1])+int8_t(receve[2]))/8192*360);
     getPitch=float((256*int8_t(receve[3])+int8_t(receve[4]))/8192*360);
     getSpeed=float((256*int8_t(receve[5])+int8_t(receve[6]))/100);
    if(receve[7]==0x11){
         getState=1;
    }
    else if(receve[7]=0x22){
         getState=2;
    }
    cout<<getYaw<<endl;
    cout<<getPitch<<endl;
    cout<<getSpeed<<endl;
    cout<<getState<<endl;
//            printf("recv false");
    close(fd);

    return true;
}
*/

/**
 * @brief port_OPEN
 * @param port
 * @return
 */
int serial::OPEN(char *port){

    //可读写，不作为终端，无时延
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    //printf("%d\n",fd);
    if (fd < 0) {
        printf("open false\n");
        return -1;
    }
    //if(fcntl(fd, F_SETFL, 0)<0)
    //printf("open success");
    if(fcntl(fd, F_SETFL, 0) < 0)
     {
         printf("fcntl failed!\n");
         return -1;
     }
     else
     {
         printf("Open Serial Port Succeed!\n");
     }
    return fd;

}
/*
    串口初始化
*/

bool serial::SET(int fd) {
    struct termios options;

    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    //修改控制模式，保证程序不会占用串口
    options.c_cflag |= CLOCAL;
    //修改控制模式，使得能够从串口中读取输入数据
    options.c_cflag |= CREAD;

    options.c_cflag &= ~CRTSCTS;//不使用流控制
    options.c_cflag &= ~PARENB;//无奇偶校验位
    options.c_iflag &= ~INPCK;
    options.c_cflag &= ~CSTOPB;//1停止位
    options.c_cflag &= ~CSIZE;//屏蔽
    options.c_cflag |= CS8;//设置数据位
    options.c_cc[VMIN] = 0;//
    options.c_cc[VTIME] = 0;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    tcflush(fd,TCIFLUSH);
    tcsetattr(fd, TCSANOW, &options);//将修改后的termios设置到串口中
    if (tcgetattr(fd, &options)!=0) {
        printf("set false\n");
        return false;
    }
    return true;
}
/*
    发送数据
*/

int serial::SEND(int fd, send_data data)
{

    int len = write(fd, data.data_all, 10);
    if (len == 10) {
        printf("send success\n");
        return len;
    }
    return 0;
}
/// <summary>
/// 发送数据
/// </summary>
/// <param name="fd">串口号</param>
/// <param name="pitch">pitch轴数据</param>
/// <param name="yaw">yaw轴数据</param>
/// <param name="mode">是否开火指令，需要开火发送0x77，否则发送0x00</param>
/// <returns></returns>
int serial::port_SEND(int &fd, int point, int mod){
       send_data data;
       Sendmode.I_mode=mod;
       SendPoint.I_Point=point;
//Yaw
        data.data_all[1]=Sendmode.U_mode[0];
        data.data_all[2]=Sendmode.U_mode[1];
        data.data_all[3]=Sendmode.U_mode[2];
        data.data_all[4]=Sendmode.U_mode[3];

//Pitch
        data.data_all[5]=SendPoint.U_Point[0];
        data.data_all[6]=SendPoint.U_Point[1];
        data.data_all[7]=SendPoint.U_Point[2];
        data.data_all[8]=SendPoint.U_Point[3];

        cout<<"data_send:";

//       for(int i=0;i<12;i++)
//       printf("%d,",data.data_all[i]);
//       cout<<endl;


//       int yaw_test=(int)((data.data_all[4]<<24)+(data.data_all[3]<<16)+(data.data_all[2]<<8)+data.data_all[1]);
//       cout<<"yaw_test:"<<yaw_test<<endl;


       int is_send=SEND(fd, data);
       if (!is_send){
           printf("send false\n");
           return false;
       }
       return is_send;
}




bool serial::port_RECV(int &fd)
{
    u_char receve[3];//暂时缓存接收到的数据
    fd_set rd_fd;
    FD_ZERO(&rd_fd);
    FD_SET(fd,&rd_fd);

    struct timeval times;
    times.tv_sec=0;
    times.tv_usec=500;

    int rd_read;
    int sel=select(fd+1,&rd_fd,NULL,NULL,&times);//times设置的时间内无接收返回0,有接收返回正数
    printf("sel:%d\n",sel);

    if(sel)//判断是否设置成功
    {
        u_char temp_data[7]={0};
        rd_read=read(fd,temp_data,7);
        printf("receve=%d\n",rd_read);
        for(int recv_i=0;recv_i<7;recv_i++){
            if(temp_data[recv_i]==0x78&&temp_data[recv_i+2]==0x76){

                for(int temp_i=0;temp_i<7;temp_i++){//把符合条件的存到类里
                    receve[temp_i]=temp_data[temp_i+recv_i];
                }
            }
        }
    }
    else return false;

    //接收协议
    if(receve[0]==0x78&&receve[2]==0x76){
        Getpanda=receve[1];
        return true;
    }

      else
      {cout<<"Received successfully, but protocol error"<<endl;
          return false;
      }
}




