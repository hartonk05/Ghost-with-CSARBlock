#pragma once
#define L 3
#define ETA (0.002)                                                                                                                                                                                                                                                                                                                                                                                                    
#define BETA (0.04)                                                                                                                                                                                                                                                                                                                                                                                                    
#define RELU 1 //中間層までの活性化関数

#define NETMODEL 4 //0:MLP  1:CNN 2:Skip 3:GhostNet 4:Ghost+CSAR

#define BATCHLEARN 1  //Skip時:0
#define BS 40 //バッチサイズ
#define EPS 0.0000001
#define MRATE 0.9 //移動平均時の慣性

// ラベルスムージング
#define LS 1		// 0:OFF, 1:ON
#define LSEPS 0.0045// パラメータ

//データのタイプに依存するもの
//#define K 1 // KAN用出力クラス数
//#define IND 1 // 入力データの次元
//#define NMAX 100 //KAN用
//#define ICH 1
//#define H 1 //KAN用
//#define W 1 //KAN用

//#define K 10 // optdigit出力クラス数
//#define IND 64 // 入力データの次元
//#define NMAX 100 //optdigit
//#define ICH 1
//#define H 8 //optdigit
//#define W 8 //optdigit
//#define IKS 1 //1*1 no-padding

//#define K 10 // MNIST出力クラス数
//#define IND 784 // 入力データの次元
//#define NMAX 784 //MNIST
//#define ICH 1
//#define H 28 //MNIST
//#define W 28 //MNIST
//#define IKS 5 //5*5 no-padding

#define K 10 // CIFAR10出力クラス数
#define IND 3072 // 入力データの次元
#define NMAX 3072 //CIFAR10
#define ICH 3
#define H 32 //CIFAR10
#define W 32 //CIFAR10
#define IKS 1 //1*1 no-padding

void BatchNormalF(double bnet[], double anet[], double* mean, double* var, double gamma, double beta);
void BatchNormalB(double delta[], double bnet[], double mean, double var, double* gamma, double* beta, double lr);

void Forward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]);
void BForward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX],
	double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double mmean[][NMAX], double mvar[][NMAX]);
void BackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double tk[], double delta[][NMAX], double lr);
void BBackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double tk[][K], double delta[][NMAX][BS],
	double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double lr);