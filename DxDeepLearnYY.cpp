#define _USE_MATH_DEFINES
#include "DxLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "MLPyy.h"
#include "CNNyy.h"

#define MENUX 120	//メニューのX, Y座標
#define MENUY 40
#define BK 0
#define WH 16777215
#define RD H

#define EPOC 10			//学習回数
#define LOOP 5			//学習回数
#define CDECAY 1		//cos decay
#define WARMUP 1		// ウォームアップ(Epoc数)
#define WARMUPRATE 0.4	// ウォームアップの係数

//#define TRA 3823
//#define TES 1797
//#define MAXV (16.0)
//char TrainFileName[] = "optdigits_tra.csv", TestFileName[] = "optdigits_tes.csv";
//int NodeN[L] = { IND,NMAX,K };
//int ChN[SEG][CONV + 1] = { { ICH,CHMAX / 4,CHMAX / 2 }, { CHMAX / 2,CHMAX / 2,CHMAX } };
//int SkipChN[SEG][CONV + 1] = { { CHMAX / 2,CHMAX / 2,CHMAX / 2 }, { CHMAX,CHMAX,CHMAX } }; //(てきとう)

//#define TRA 30000 //30000
//#define TES 10000 //10000
//#define MAXV (255.0)
//char TrainFileName[] = "dataset_mnist_train.csv", TestFileName[] = "dataset_mnist_test.csv";
//int NodeN[L] = { IND,160,K };
////書ききれていない
////int ChN[SEG][CONV + 1] = { { ICH,CHMAX / 4,CHMAX / 2 }, { CHMAX / 2,CHMAX / 2,CHMAX } };
////int SkipChN[SEG][CONV + 1] = { { CHMAX / 2,CHMAX / 2,CHMAX / 2 }, { CHMAX,CHMAX,CHMAX } };
//int ChN[SEG][CONV + 1] = { { ICH,CHMAX / 4,CHMAX / 4 }, { CHMAX / 4,CHMAX / 2,CHMAX / 2 }, { CHMAX / 2,CHMAX,CHMAX } };
//int SkipChN[SEG][CONV + 1] = { { CHMAX / 4,CHMAX / 4,CHMAX / 4 },{ CHMAX / 2,CHMAX / 2,CHMAX / 2 },{ CHMAX,CHMAX,CHMAX } };
////int SkipChN[SEG][CONV + 1] = { { CHMAX/8,CHMAX/8,CHMAX/8 },{ CHMAX / 4,CHMAX / 4,CHMAX / 4 },{ CHMAX / 2,CHMAX / 2,CHMAX / 2 } };

#define TRA 500 //50000
#define TES 200  //10000
#define MAXV (255.0)
char TrainFileName[] = "cifar10_train.csv", TestFileName[] = "cifar10_test.csv";
int NodeN[L] = { IND,200,K };
int ChN[SEG][CONV + 1] = { { ICH,CHMAX / 4,CHMAX / 4 }, { CHMAX / 4,CHMAX / 2,CHMAX / 2 }, { CHMAX / 2,CHMAX,CHMAX } };
int SkipChN[SEG][CONV + 1] = { { CHMAX / 4,CHMAX / 4,CHMAX / 4 },{ CHMAX / 2,CHMAX / 2,CHMAX / 2 },{ CHMAX,CHMAX,CHMAX } };
//int ChN[SEG][CONV + 1] = { { ICH,CHMAX / 8,CHMAX / 8 }, { CHMAX / 8,CHMAX / 4,CHMAX/4 }, { CHMAX / 4,CHMAX / 2,CHMAX } };
//int SkipChN[SEG][CONV + 1] = { { CHMAX / 8,CHMAX / 8,CHMAX / 8 },{ CHMAX / 4,CHMAX / 4,CHMAX / 4 },{ CHMAX,CHMAX,CHMAX } };

unsigned char TraData[TRA][IND], TesData[TES][IND];
int TraClass[TRA], TesClass[TES], TraNum[TRA];
int In, Epoc;
double TrainErr, TestErr, TrainAcc, TestAcc, Lrate;
time_t Start, End;

// ファイル保存名
char SaveFileName[] = "AccLogRun.csv";

double Out[L][NMAX], Tk[K], Delta[L][NMAX];
double Bias[L][NMAX], Woi[L - 1][NMAX][NMAX]; //NNの重み

double BOut[L][BS][NMAX], BTk[BS][K], BDelta[L][NMAX][BS];
//バッチ正則化
double BNet[L][NMAX][BS], Mean[L][NMAX], Var[L][NMAX];
double Gamma[L][NMAX], Beta[L][NMAX], MMean[L][NMAX], MVar[L][NMAX];

//CNN 用
//int ChN[SEG][CONV + 1] = { { ICH,CHMAX / 2}, { CHMAX / 2,CHMAX } };
double COut[SEG][CONV + 1][CHMAX][H][W], CDelta[SEG][CONV + 1][CHMAX][H][W];
double CBias[SEG][CONV][CHMAX], CWoi[SEG][CONV][CHMAX][CHMAX][KS][KS]; //NNの重み
double DCBias[SEG][CONV][CHMAX], DCWoi[SEG][CONV][CHMAX][CHMAX][KS][KS]; //修正量を保持する変数
double BCOut[SEG][CONV + 1][CHMAX][BS][H][W], BCDelta[SEG][CONV + 1][CHMAX][BS][H][W];
//CNNバッチ正則化
double CBNet[SEG][CONV + 1][CHMAX][BS][H][W], CMean[SEG][CONV + 1][CHMAX], CVar[SEG][CONV + 1][CHMAX];
double CGamma[SEG][CONV + 1][CHMAX], CBeta[SEG][CONV + 1][CHMAX], CMMean[SEG][CONV + 1][CHMAX], CMVar[SEG][CONV + 1][CHMAX];

//Skip-Connection
double InOut[ICH][H][W], InBias[CHMAX], InWoi[CHMAX][ICH][IKS][IKS];
double DInBias[CHMAX], DInWoi[CHMAX][ICH][IKS][IKS];
double SkipOut[SEG][CHMAX][H][W], SkipDelta[SEG][CHMAX][H][W];
double SkipBias[SEG][CHMAX], SkipWoi[SEG][CHMAX][CHMAX][KS][KS]; //NNの重み
double DSkipBias[SEG][CHMAX], DSkipWoi[SEG][CHMAX][CHMAX][KS][KS]; //NNの重み
double BInOut[ICH][BS][H][W], BSkipOut[SEG][CHMAX][BS][H][W], BSkipDelta[SEG][CHMAX][BS][H][W];
//SkipCNNバッチ正則化
//double BSkipNet[SEG][CONV + 1][CHMAX][BS][H][W], CMean[SEG][CONV + 1][CHMAX], CVar[SEG][CONV + 1][CHMAX];
//double CGamma[SEG][CONV + 1][CHMAX], CBeta[SEG][CONV + 1][CHMAX], CMMean[SEG][CONV + 1][CHMAX], CMVar[SEG][CONV + 1][CHMAX];

//GBottleneckバッチ正則化
double GBias[SEG][CONV][CHMAX], GWoi[SEG][CONV][CHMAX][CHMAX][KS][KS], GIn[SEG][CONV][CHMAX][H][W], GOut[SEG][CONV][CHMAX][H][W];
double DGBias[SEG][CONV][CHMAX], DGWoi[SEG][CONV][CHMAX][CHMAX][KS][KS], GDelta[SEG][CONV][CHMAX][H][W];
double GMean[SEG][CONV][CHMAX], GVar[SEG][CONV][CHMAX], GGamma[SEG][CONV][CHMAX], GBeta[SEG][CONV][CHMAX], GMMean[SEG][CONV][CHMAX], GMVar[SEG][CONV][CHMAX];
//double GInDelta[SEG][CONV][CHMAX][H][W], BGInDelta[SEG][CONV][CHMAX][BS][H][W];	//CSAR時に追加
double BGNet[SEG][CONV][CHMAX][BS][H][W], BGOut[SEG][CONV][CHMAX][BS][H][W], BGIn[SEG][CONV][CHMAX][BS][H][W];// , BGDelta[SEG][CONV][CHMAX][BS][H][W];

//提案手法
// 1*1CONV
//CAUnit用
double CABias[SEG][CONV][2][CHMAX], CAWoi[SEG][CONV][2][CHMAX][CHMAX], CAOut[SEG][CONV][2][CHMAX][H][W], CUnitOut[SEG][CONV][CHMAX][H][W], GAPIn[SEG][CONV][CHMAX][H][W], CAGamma[SEG][CONV][2][CHMAX], CABeta[SEG][CONV][2][CHMAX], CAMMean[SEG][CONV][2][CHMAX], CAMVar[SEG][CONV][2][CHMAX];
double DCABias[SEG][CONV][2][CHMAX], DCAWoi[SEG][CONV][2][CHMAX][CHMAX];// , CUnitDelta[SEG][CONV][CHMAX][H][W];
//SAUnit用
double SABias[SEG][CONV][2][CHMAX], SAWoi[SEG][CONV][2][CHMAX][CHMAX], SAOut[SEG][CONV][2][CHMAX][H][W], SUnitOut[SEG][CONV][CHMAX][H][W], SAGamma[SEG][CONV][2][CHMAX], SABeta[SEG][CONV][2][CHMAX], SAMMean[SEG][CONV][2][CHMAX], SAMVar[SEG][CONV][2][CHMAX];
double DSABias[SEG][CONV][2][CHMAX], DSAWoi[SEG][CONV][2][CHMAX][CHMAX];// , SUnitDelta[SEG][CONV][CHMAX][H][W];
//CSAR Block用
double ResBias[SEG][CONV][CHMAX], ResWoi[SEG][CONV][CHMAX][CHMAX*2], ResOut[SEG][CONV][CHMAX][H][W];
double DResBias[SEG][CONV][CHMAX], DResWoi[SEG][CONV][CHMAX][CHMAX*2];

//Ghost+CSARバッチ正則化
double BCAOut[SEG][CONV][2][CHMAX][BS][H][W], BCUnitOut[SEG][CONV][CHMAX][BS][H][W], BGAPIn[SEG][CONV][CHMAX][BS][H][W];
double BSAOut[SEG][CONV][2][CHMAX][BS][H][W], BSUnitOut[SEG][CONV][CHMAX][BS][H][W], BResOut[SEG][CONV][CHMAX][BS][H][W];
//double BCUnitDelta[SEG][CONV][CHMAX][BS][H][W], BSUnitDelta[SEG][CONV][CHMAX][BS][H][W];

void Display();
void ReadFileData();
void Initialize();
void Input(unsigned char indata[], double out[]);
void CInput(unsigned char indata[], double out[][H][W]);
void MLP();
void STest();

//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_  HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd) {
	int MouseInput, MouseX, MouseY, k,i;

	//srand((unsigned int)time(NULL));
	srand(1);
	ReadFileData();//学習データ準備
	Initialize();//NN初期化

	SetGraphMode(1200, 800, 32);	//画面モードの設定	
	SetBackgroundColor(255, 255, 255);
	ChangeWindowMode(TRUE);			//ウインドウモードに変更
	SetAlwaysRunFlag(TRUE);			//バックグラウンドでも実行を継続
	SetMouseDispFlag(TRUE);			//マウスを表示状態にする	
	if (DxLib_Init() == -1) return -1;	// ＤＸライブラリ初期化処理 エラーが起きたら直ちに終了

	Display();
	while (1) {
		if (ProcessMessage() == -1) break;	//エラーが起きたらループから抜ける
		MouseInput = GetMouseInput();	//マウスの入力を待つ			
		if ((MouseInput & MOUSE_INPUT_LEFT) != 0) {					//左ボタン押された
			GetMousePoint(&MouseX, &MouseY);						//マウスの位置を取得
			if (MouseX < MENUX) { // Menu area click
				if (MouseY < MENUY) break;				//END
				else if (MouseY < MENUY * 2) Initialize();//NN初期化
				else if (MouseY < MENUY * 3) { //Input
					In = (rand() / (RAND_MAX + 1.0)) * TRA;
					if (NETMODEL == 1) {
						CInput(TraData[In], COut[0][0]);
						CForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar);
					}
					else Input(TraData[In], Out[0]);
					for (k = 0; k < K; k++) Tk[k] = 0;
					Tk[TraClass[In]] = 1.0;//Hot One Vecter
				}
				else if (MouseY < MENUY * 4)
					Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta); //移動平均あり
				else if (MouseY < MENUY * 5) MLP(); //Learn
				else if (MouseY < MENUY * 6) {
					for (i = 0; i < LOOP; i++) {
						Initialize();
						MLP();
					}
				}
				//else if (MouseY < MENUY * 7) {
				//}
			}
			Display();
		}
		WaitTimer(100);
	}
	DxLib_End();				//ＤＸライブラリ使用の終了処理
	return 0;					//ソフトの終了 
}
void Display() {
	int i, j, k, mx, my, cr, size = 2, b;
	char menu[][20] = { "END", "Init", "Input", "Forward", "Learn","N-Learn" };

	ClearDrawScreen();
	for (i = 0; i < 6; i++) {	//メニュー表示
		DrawString(10, i * MENUY + 10, menu[i], BK);
		DrawLine(0, MENUY * (i + 1), MENUX, MENUY * (i + 1), BK, 2);
	}
	DrawLine(MENUX, 0, MENUX, MENUY * 6, BK, 2);

	if (NETMODEL == 1) DrawString(10, MENUY * 6 + 10, "CNN", BK);
	else if (NETMODEL == 2) DrawString(10, MENUY * 6 + 10, "SkipCNN", BK);
	else if (NETMODEL == 3) DrawString(10, MENUY * 6 + 10, "Ghost", BK);
	else if (NETMODEL == 4) DrawString(10, MENUY * 6 + 10, "Ghost+CSAR", BK);
	else DrawString(10, MENUY * 6 + 10, "MLP", BK);
	//if(BATCHLEARN) DrawFormatString(10, MENUY * 7+24, BK, "Learning Rate %.3lf", Lrate);
	//else 
	DrawFormatString(10, MENUY * 7 + 24, BK, "Learning Rate %.3lf", Lrate);
	DrawFormatString(10, MENUY * 8 + 10, BK, "Epoc %3d/%3d   Time %3d", Epoc, EPOC, End - Start);
	DrawFormatString(10, MENUY * 9, BK, "TraAcc %.3lf  %d", TrainAcc, TRA);
	DrawFormatString(10, MENUY * 9 + 24, BK, "TesAcc %.3lf  %d", TestAcc, TES);

	mx = MENUX + 100; my = MENUY;
	for (i = 0; i < RD; i++) for (j = 0; j < RD && (i * RD + j) < NodeN[0]; j++) {//全結合層入力データの情報
		if (Out[0][i * RD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
		else {
			cr = (int)(Out[0][i * RD + j] * 255);
			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
		}
	}
	for (k = 0; k < K; k++) if (Tk[k] > 0) DrawFormatString(mx + 58, my + 50, BK, "%d", k);


	for (k = 0; k < K; k++) // 出力の表示
		DrawFormatString(mx, my + size * RD + k * 20, BK, "%.2lf", Out[L - 1][k]);
	if (NETMODEL == 1 || NETMODEL == 2 || NETMODEL == 3 || NETMODEL == 4) {//畳み込み層入力データの情報
		for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
			if (COut[0][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				//cr = (int)(COut[0][0][0][i][j] * 255);
				//DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				cr = GetColor((int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255));
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
			}
		}
		for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
			if (COut[0][1][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(COut[0][1][0][i][j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		for (my = 500, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
			if (COut[1][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(COut[1][0][0][i][j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		if (SEG > 2)
			for (my = 600, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
				if (COut[2][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
				else {
					cr = (int)(COut[2][0][0][i][j] * 255);
					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				}
			}
	}

	if (BATCHLEARN) {
		DrawFormatString(10, MENUY * 7, BK, "BatchLearn %d", BS);
		for (mx = MENUX + 200, b = 0; b < BS && b < 10; b++, mx += 80) {
			for (my = MENUY, i = 0; i < RD; i++) for (j = 0; j < RD && (i * RD + j) < NodeN[0]; j++) {//全結合層入力データの情報
				if (BOut[0][b][i * RD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
				else {
					cr = (int)(BOut[0][b][i * RD + j] * 255);
					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				}
			}
			for (k = 0; k < K; k++) if (BTk[b][k] > 0) DrawFormatString(mx + 58, my + 50, BK, "%d", k);
			for (k = 0; k < K; k++) // 出力の表示
				DrawFormatString(mx, my + size * RD + k * 20, BK, "%.2lf", BOut[L - 1][b][k]);
			if (NETMODEL == 1 || NETMODEL == 2 || NETMODEL == 3 || NETMODEL == 4) {//畳み込み層入力データの情報
				for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
					if (BCOut[0][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
					else {
						cr = (int)(BCOut[0][0][0][b][i][j] * 255);
						DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
					}
				}
				for (my = 380, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
					if (BCOut[0][1][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
					else {
						cr = (int)(BCOut[0][1][0][b][i][j] * 255);
						DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
					}
				}
				for (my = 460, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
					if (BCOut[1][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
					else {
						cr = (int)(BCOut[1][0][0][b][i][j] * 255);
						DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
					}
				}
				//if (SEG > 2)
				//	for (my = 540, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
				//		if (BCOut[2][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
				//		else {
				//			cr = (int)(BCOut[2][0][0][b][i][j] * 255);
				//			DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				//		}
				//	}
			}
		}
	}
	ScreenFlip(); WaitTimer(10);
}

void ReadFileData() {
	FILE* fp;
	int i, n;
	fopen_s(&fp, TrainFileName, "r");
	for (n = 0; n < TRA; n++) {
		for (i = 0; i < IND; i++) {
			//fscanf_s(fp, "%lf,", &TraData[n][i]);
			//TraData[n][i] /= MAXV;
			fscanf_s(fp, "%d,", &TraData[n][i]);
		}
		fscanf_s(fp, "%d\n", &TraClass[n]);
	}
	fclose(fp);

	/* -------- テストデータ読み込み -------- */
	fopen_s(&fp, TestFileName, "r");
	for (n = 0; n < TES; n++) {
		for (i = 0; i < IND; i++) {
			//fscanf_s(fp, "%lf,", &TesData[n][i]);
			//TesData[n][i] /= MAXV;
			fscanf_s(fp, "%d,", &TesData[n][i]);
		}
		fscanf_s(fp, "%d\n", &TesClass[n]);
	}
	fclose(fp);
}
void Initialize() {// 重みとバイアスの初期値を与える
	int i, j, l, s, c, y, x;
	double he_wp, och;
	if (NETMODEL == 1) NodeN[0] = (H / pow(PS, SEG)) * (W / pow(PS, SEG)) * CHMAX;
	else if (NETMODEL == 2 || NETMODEL == 3 || NETMODEL == 4) NodeN[0] = ((H - IKS + 1) / pow(PS, SEG)) * ((W - IKS + 1) / pow(PS, SEG)) * SkipChN[SEG - 1][CONV];
	//else if (NETMODEL == 2) NodeN[0] = ((H - IKS + 1) / pow(PS, SEG)) * ((W - IKS + 1) / pow(PS, SEG)) * CHMAX;
	//else if (NETMODEL == 3) NodeN[0] = SkipChN[SEG-1][CONV]; //GAP
	else NodeN[0] = IND;

	for (l = 0; l < L; l++)for (i = 0; i < NMAX; i++) {
		Gamma[l][i] = 1.0; Beta[l][i] = 0;
		MMean[l][i] = 0; MVar[l][i] = 1;
	}
	if (RELU) { //He初期化
		for (l = 0; l < L - 2; l++) {//中間層外側　ReLUする場合もある
			he_wp = 2 * sqrt(6.0 / NodeN[l]);
			for (j = 0; j < NodeN[l + 1]; j++)//出力側
				for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //入力側
					Woi[l][j][i] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp;
		}
	}
	else
		for (l = 0; l < L - 2; l++) //中間層外側　ReLUする場合もある
			for (j = 0; j < NodeN[l + 1]; j++)//出力側
				for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //入力側
					Woi[l][j][i] = (double)rand() / (RAND_MAX + 1.0) - 0.5; //+-0.5

	for (j = 0; j < NodeN[l + 1]; j++)//出力側
		for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //入力側
			Woi[l][j][i] = (double)rand() / (RAND_MAX + 1.0) - 0.5;

	if (NETMODEL == 1) { //CNN 重み初期化 ReLU
		for (s = 0; s < SEG; s++) for (c = 0; c <= CONV; c++) for (j = 0; j < CHMAX; j++) {
			CGamma[s][c][j] = 1.0; CBeta[s][c][j] = 0;
			CMMean[s][c][j] = 0; CMVar[s][c][j] = 1;
		}

		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (ChN[s][c] * KS * KS)), j = 0; j < ChN[s][c + 1]; j++)
				for (CBias[s][c][j] = 0, i = 0; i < ChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						CWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
	}
	if (NETMODEL == 2 || NETMODEL == 3 || NETMODEL == 4) { //skipCon 重み初期化 ReLU
		for (he_wp = 2 * sqrt(6.0 / (ICH * IKS * IKS)), j = 0; j < SkipChN[0][0]; j++)
			for (InBias[j] = 0, i = 0; i < ICH; i++)
				for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
					InWoi[j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		for (s = 0; s < SEG; s++) {
			if (s == SEG - 1)och = CHMAX;
			else och = SkipChN[s + 1][0];
			//for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][CONV] * KS * KS)), j = 0; j < och; j++)
				for (SkipBias[s][j] = 0, i = 0; i < SkipChN[s][CONV]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						SkipWoi[s][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		}
		// バッチ正規化
		for (s = 0; s < SEG; s++) for (c = 0; c <= CONV; c++)for (j = 0; j < CHMAX; j++) {
			CGamma[s][c][j] = 1.0; CBeta[s][c][j] = 0;
			CMMean[s][c][j] = 0; CMVar[s][c][j] = 1;
		}
		// 重み・バイアス
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (CBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						CWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
	}
	if (NETMODEL == 3 || NETMODEL == 4) {//Ghost Bottleneck
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (GBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						GWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		// バッチ正規化
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (j = 0; j < CHMAX; j++) {
			GGamma[s][c][j] = 1.0; GBeta[s][c][j] = 0;
			GMMean[s][c][j] = 0; GMVar[s][c][j] = 1;
		}
	}
	if (NETMODEL == 4) {
		//CAUnit
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (l = 0; l < 2; l++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (CABias[s][c][l][j] = 0, i = 0; i < SkipChN[s][c+1]; i++)
						CAWoi[s][c][l][j][i] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		// バッチ正規化
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (l = 0; l < 2; l++)for (j = 0; j < CHMAX; j++) {
			CAGamma[s][c][l][j] = 1.0; CABeta[s][c][l][j] = 0;
			CAMMean[s][c][l][j] = 0; CAMVar[s][c][l][j] = 1;
		}
		//SAUnit
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (l = 0; l < 2; l++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (SABias[s][c][l][j] = 0, i = 0; i < SkipChN[s][c]; i++)
						SAWoi[s][c][l][j][i] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		// バッチ正規化
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (l = 0; l < 2; l++)for (j = 0; j < CHMAX; j++) {
			SAGamma[s][c][l][j] = 1.0; SABeta[s][c][l][j] = 0;
			SAMMean[s][c][l][j] = 0; SAMVar[s][c][l][j] = 1;
		}
		//CS統合
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c]*2 * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (ResBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]*2; i++)
					ResWoi[s][c][j][i] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
	}
}
void Input(unsigned char indata[], double out[]) {
	int i;
	for (i = 0; i < IND; i++) out[i] = indata[i] / MAXV;
}
void CInput(unsigned char indata[], double out[][H][W]) {
	int i, h, w, n;
	for (n = 0, i = 0; i < ICH; i++)
		for (h = 0; h < H; h++)for (w = 0; w < W; w++, n++)
			//out[i][h][w] = indata[i*H*W+h*W+w] / MAXV;
			out[i][h][w] = indata[n] / MAXV;
}
void MLP() {
	int d, in, k, count, maxk, b, i, temp, h, w;
	double ceta;

	FILE* fp;

	if (BATCHLEARN) {
		if (NETMODEL == 2 && BATCHLEARN) fopen_s(&fp, "LogBFile\\Skip-CNN_Batch.csv", "a");
		else if (NETMODEL == 3 && BATCHLEARN) fopen_s(&fp, "LogBFile\\GNet_Batch.csv", "a");
		else if (NETMODEL == 4 && BATCHLEARN) fopen_s(&fp, "LogBFile\\GNet_CSAR_Batch.csv", "a");
		else fopen_s(&fp, SaveFileName, "a");
	}
	else {
		if (NETMODEL == 2) fopen_s(&fp, "LogFile\\Skip-CNN.csv", "a");
		else if (NETMODEL == 3) fopen_s(&fp, "LogFile\\GNet.csv", "a");
		else if (NETMODEL == 4) fopen_s(&fp, "LogFile\\GNet_CSAR.csv", "a");
		else fopen_s(&fp, SaveFileName, "a");
	}
	fprintf_s(fp, "\nEpoc,TrainAcc,TestAcc,Time\n");

	Start = time(NULL);
	for (Epoc = 0; Epoc < EPOC; Epoc++) { // Train data 一通り学習
		for (i = 0; i < TRA; i++)TraNum[i] = i;
		for (i = 0; i < TRA; i++) {
			in = (rand() / (RAND_MAX + 1.0)) * (TRA - i) + i;
			temp = TraNum[i];
			TraNum[i] = TraNum[in];
			TraNum[in] = temp;
		}
		if (BATCHLEARN)ceta = BETA; else ceta = ETA;
		if (CDECAY) {
			if (Epoc < WARMUP) Lrate = ceta * WARMUPRATE;
			else Lrate = 0.5 * (1 + cos(((Epoc - WARMUP) * M_PI) / (EPOC))) * ceta;
		}
		else Lrate = ceta;

		if (BATCHLEARN) {
			for (count = 0, d = 0; d < TRA - BS; d += BS) {
				for (b = 0; b < BS; b++) {
					//in = (rand() / (RAND_MAX + 1.0)) * TRA;
					in = TraNum[d + b];
					if (NETMODEL == 1) {//CInput(TraData[in], BCOut[0][0][b]);
						for (k = 0, i = 0; i < ICH; i++)
							for (h = 0; h < H; h++)for (w = 0; w < W; w++, k++)
								BCOut[0][0][i][b][h][w] = TraData[in][k]/MAXV;
					}
					else if (NETMODEL == 2 || NETMODEL == 3 || NETMODEL == 4) {
						for (k = 0, i = 0; i < ICH; i++)
							for (h = 0; h < H; h++)for (w = 0; w < W; w++, k++)
								BInOut[i][b][h][w] = TraData[in][k] / MAXV;
					}
					else Input(TraData[in], BOut[0][b]);
					/* 教師データ作成 */
					if (LS == 1) {// ラベル平滑化
						for (k = 0; k < K; k++) BTk[b][k] = LSEPS / (K - 1);
						BTk[b][TraClass[in]] = 1.0 - LSEPS;//Hot One Vecter
					}
					else {
						for (k = 0; k < K; k++) BTk[b][k] = 0;
						BTk[b][TraClass[in]] = 1.0;//Hot One Vecter
					}
				}
				if (NETMODEL == 1)BCForward(ChN, CBias, CWoi, BCOut, BOut[0], CBNet, CMean, CVar, CGamma, CBeta, CMMean, CMVar);
				if (NETMODEL == 2)BSkipForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CBNet, CMean, CVar, CGamma, CBeta, CMMean, CMVar, BInOut, InBias, InWoi, BSkipOut, SkipBias, SkipWoi);
				if (NETMODEL == 3)BGForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CBNet, CMean, CVar, CGamma, CBeta, CMMean, CMVar, BInOut, InBias, InWoi, BSkipOut, SkipBias, SkipWoi, GBias, GWoi, BGIn, BGOut, BGNet, GMean, GVar, GGamma, GBeta, GMMean, GMVar);
				if (NETMODEL == 4)BCSGForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CBNet, CMean, CVar, CGamma, CBeta, CMMean, CMVar, BInOut, InBias, InWoi, BSkipOut, SkipBias, SkipWoi, GBias, GWoi, BGIn, BGOut, BGNet,GMean, GVar, GGamma, GBeta, GMMean, GMVar, 
					CABias, CAWoi, BCAOut, CAGamma, CABeta, CAMMean, CAMVar, BCUnitOut, BGAPIn, SABias, SAWoi, BSAOut, SAGamma, SABeta, SAMMean, SAMVar, BSUnitOut, ResBias, ResWoi, BResOut);
				BForward(NodeN, Bias, Woi, BOut, BNet, Mean, Var, Gamma, Beta, MMean, MVar);
				BBackProp(NodeN, Bias, Woi, BOut, BTk, BDelta, BNet, Mean, Var, Gamma, Beta, Lrate);
				if (NETMODEL == 1) BCBackProp(ChN, CBias, CWoi, BCOut, BCDelta, BDelta[0], DCBias, DCWoi, Lrate, CBNet, CMean, CVar, CGamma, CBeta);
				if (NETMODEL == 2) BSkipBackProp(SkipChN, CBias, CWoi, BCOut, BCDelta, BDelta[0], DCBias, DCWoi, Lrate, CBNet, CMean, CVar, CGamma, CBeta, BInOut, InBias, InWoi, BSkipOut, BSkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi);
				if (NETMODEL == 3) BGBackProp(SkipChN, CBias, CWoi, BCOut, BCDelta, BDelta[0], DCBias, DCWoi, Lrate, CBNet, CMean, CVar, CGamma, CBeta, BInOut, InBias, InWoi, BSkipOut, BSkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
					BGIn, BGOut, GBias, GWoi, DGBias, DGWoi, BGNet, GMean, GVar, GGamma, GBeta);
				if (NETMODEL == 4) BCSGBackProp(SkipChN, CBias, CWoi, BCOut, BCDelta, BDelta[0], DCBias, DCWoi, Lrate, CBNet, CMean, CVar, CGamma, CBeta, BInOut, InBias, InWoi, BSkipOut, BSkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
					BGIn, BGOut, GBias, GWoi, DGBias, DGWoi, BGNet, GMean, GVar, GGamma, GBeta,
					CABias, CAWoi, BCAOut, BCUnitOut, BGAPIn, SABias, SAWoi, BSAOut, BSUnitOut, ResBias, ResWoi, BResOut,
					CAGamma, CABeta, CAMMean, CAMVar, SAGamma, SABeta, SAMMean, SAMVar, DCABias, DCAWoi,DSABias, DSAWoi, DResBias, DResWoi);

				for (b = 0; b < BS; b++) {
					for (maxk = 0, k = 1; k < K; k++)
						if (BOut[L - 1][b][maxk] < BOut[L - 1][b][k]) maxk = k;
					//if (BTk[b][maxk] == 1.0)count++;
					if (BTk[b][maxk] > 0.5)count++;
				}
			}
		}
		else {
			for (count = 0, d = 0; d < TRA; d++) {
				//in = (rand() / (RAND_MAX + 1.0)) * TRA;
				in = TraNum[d];
				if (NETMODEL == 1) {
					CInput(TraData[in], COut[0][0]);
					CForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar);
				}
				else if (NETMODEL == 2) {
					CInput(TraData[in], InOut);
					SkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);
				}
				else if (NETMODEL == 3) {
					CInput(TraData[in], InOut);
					GForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi, GBias, GWoi, GIn, GOut, GGamma, GBeta, GMMean, GMVar);//あり
				}
				else if (NETMODEL == 4) {
					CInput(TraData[in], InOut);
					CSGForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi, GBias, GWoi, GIn, GOut, GGamma, GBeta, GMMean, GMVar, CABias, CAWoi, CAOut, CAGamma, CABeta, CAMMean, CAMVar, CUnitOut, GAPIn, SABias, SAWoi, SAOut, SAGamma, SABeta, SAMMean, SAMVar, SUnitOut, ResBias, ResWoi, ResOut);//あり
				}
				else Input(TraData[in], Out[0]);
				/* 教師データ作成 */
				if (LS == 1) {// ラベル平滑化
					for (k = 0; k < K; k++) Tk[k] = LSEPS / (K - 1);
					Tk[TraClass[in]] = 1.0 - LSEPS;//Hot One Vecter
				}
				else {
					for (k = 0; k < K; k++) Tk[k] = 0;
					Tk[TraClass[in]] = 1.0;//Hot One Vecter
				}
				Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta);
				BackProp(NodeN, Bias, Woi, Out, Tk, Delta, Lrate);
				if (NETMODEL == 1)CBackProp(ChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Lrate);
				if (NETMODEL == 2)SkipBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Lrate, InOut, InBias, InWoi, SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi);
				if (NETMODEL == 3)GBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Lrate, InOut, InBias, InWoi, SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
					GIn, GOut, GBias, GWoi, DGBias, DGWoi);
				if (NETMODEL == 4)CSGBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Lrate, InOut, InBias, InWoi, SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
					GIn, GOut, GBias, GWoi, DGBias, DGWoi, CABias, CAWoi, CAOut, CUnitOut, GAPIn, SABias, SAWoi, SAOut, SUnitOut, ResBias, ResWoi, ResOut, DCABias, DCAWoi, DSABias, DSAWoi,DResBias, DResWoi);

				for (maxk = 0, k = 1; k < K; k++)
					if (Out[L - 1][maxk] < Out[L - 1][k]) maxk = k;
				if (maxk == TraClass[in])count++;
			}
		}
		TrainAcc = (double)count / TRA;
		End = time(NULL);
		fprintf_s(fp, "%d,%.4lf,0,%d\n", Epoc, TrainAcc, End - Start);
		Display();
	}
	// test 
	for (count = 0, d = 0; d < TES; d++) {
		in = d;
		if (NETMODEL == 1) {
			CInput(TesData[in], COut[0][0]);
			CForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar);
		}
		else if (NETMODEL == 2) {
			CInput(TesData[in], InOut);
			SkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);
		}
		else if (NETMODEL == 3) {
			CInput(TesData[in], InOut);
			//GForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);//ボトルネック無し
			GForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi, GBias, GWoi, GIn, GOut, GGamma, GBeta, GMMean, GMVar);//あり
		}
		else if (NETMODEL == 4) {
			CInput(TesData[in], InOut);
			//GForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);//ボトルネック無し
			CSGForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi, GBias, GWoi, GIn, GOut, GGamma, GBeta, GMMean, GMVar, CABias, CAWoi, CAOut, CAGamma, CABeta, CAMMean, CAMVar, CUnitOut, GAPIn, SABias, SAWoi, SAOut, SAGamma, SABeta, SAMMean, SAMVar, SUnitOut, ResBias, ResWoi, ResOut);//あり
		}
		else Input(TesData[in], Out[0]);
		Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta);
		for (maxk = 0, k = 1; k < K; k++)
			if (Out[L - 1][maxk] < Out[L - 1][k]) maxk = k;
		if (maxk == TesClass[in])count++;
	}
	TestAcc = (double)count / TES;
	fprintf_s(fp, "%d,%.4lf,%.4lf,%d\n", Epoc, TrainAcc, TestAcc, End - Start);
	fclose(fp);
}

void STest() {// test時は移動平均を利用する　（バッチなし）
	int t, k, max, count, lay = L - 1;
	FILE* testfp;
	if (fopen_s(&testfp, "testlog.csv", "w")) return;//file open 失敗
	for (TrainErr = 0, t = 0, count = 0; t < TRA; t++) { // 学習データ
		if (NETMODEL == 1) {// CNN 
			CInput(TraData[t], COut[0][0]);
			CForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar);// Out[0]を計算
		}
		else if (NETMODEL == 2) {// Skip 
			CInput(TraData[t], InOut);
			SkipForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);// Out[0]を計算
		}
		else if (NETMODEL == 3) {// Ghost 
			CInput(TraData[t], InOut);
			//GForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);// Out[0]を計算
			GForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi, GBias, GWoi, GIn, GOut, GGamma, GBeta, GMMean, GMVar);//Out[0]を計算,ボトルネックあり
		}
		else if (NETMODEL == 4) {// Ghost+CSAR
			CInput(TraData[t], InOut);
			//GForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);// Out[0]を計算
			CSGForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi, GBias, GWoi, GIn, GOut, GGamma, GBeta, GMMean, GMVar, CABias, CAWoi, CAOut, CAGamma, CABeta, CAMMean, CAMVar, CUnitOut, GAPIn, SABias, SAWoi, SAOut, SAGamma, SABeta, SAMMean, SAMVar, SUnitOut, ResBias, ResWoi, ResOut);//あり//Out[0]を計算,ボトルネックあり
		}
		else Input(TraData[t], Out[0]); // MLP only
		Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta); //移動平均あり		Forward(Out, Woi, Bias, NodeN, MAve, MVar, Gamma, Beta);
		for (max = 0, k = 1; k < K; k++)
			if (Out[lay][max] < Out[lay][k]) max = k;
		if (TraClass[t] == max) count++; 		//精度
		for (k = 0; k < K; k++)
			if (k == TraClass[t])TrainErr += (1.0 - Out[lay][k]) * (1.0 - Out[lay][k]);
		//else TrainErr += (Out[lay][k]) * (Out[lay][k]);
	}
	TrainAcc = (double)count / (double)(t); TrainErr /= (double)(t);

	for (TestErr = 0, t = 0, count = 0; t < TES; t++) { // テストデータ
		if (NETMODEL == 1) {// CNN 
			CInput(TesData[t], COut[0][0]);
			CForward(ChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar);// Out[0]を計算
		}
		else Input(TesData[t], Out[0]); // MLP only
		Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta); //移動平均あり		Forward(Out, Woi, Bias, NodeN, MAve, MVar, Gamma, Beta);
		for (max = 0, k = 1; k < K; k++)
			if (Out[lay][max] < Out[lay][k]) max = k;
		if (TesClass[t] == max) count++; 		//精度
		for (k = 0; k < K; k++)
			if (k == TesClass[t])TestErr += (1.0 - Out[lay][k]) * (1.0 - Out[lay][k]);
		//else TwatErr += (Out[lay][k]) * (Out[lay][k]);
	}
	TestAcc = (double)count / (double)(t); TestErr /= (double)(t);
	fclose(testfp);
}
