#ifndef OBJECT_H
#define OBJECT_H
#include <iostream>
#include <opencv2/opencv.hpp>
#define OPPOSIT 2           //对方代表
#define OURSIDE 3           //我方地标
#define ANYSIDE 1           //无任何一方地标
#define critical_value 8    //路线临界

using namespace cv;
/*
 * labels will be green or red or none
 * */
typedef struct object{
    Rect rect;
    std::string label;

} object;


typedef struct landmark{
    int number=0;                      //序号
    int status=0;                      //地标状态

} landmark;

typedef struct parameters{
    std::string my_team ="green";
    std::string opposide_team ="red";

//    std::string my_team ="red";
//    std::string opposide_team ="green";  6 Point(586,154)  7  Point(482,133)

    std::vector<Point> mark_confit={Point(164,234),Point(155,300),Point(271,357),Point(484,351),Point(634,284),
                                    Point(652,214),Point(586,166), Point(479,143), Point(363,149), Point(250,178)};

    std::vector<Point> marks = {Point(164,218),Point(153,286),Point(274,340),Point(486,333),
                                Point(635,268),Point(652,198),Point(586,154), Point(482,133)};                     //所有地标
} parameters;

#endif // OBJECT_H
