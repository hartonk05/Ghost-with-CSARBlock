#include<math.h>
#include"MLPyy.h"
#include"CNNyy.h"
#define LEAKY 0.0 //Leaky ReLU

double Pin[CHMAX][H + KS - 1][W + KS - 1];
double Pdelta[CHMAX][H + KS - 1][W + KS - 1];
double BPin[CHMAX][BS][H + KS - 1][W + KS - 1];
double BPdelta[CHMAX][BS][H + KS - 1][W + KS - 1];
double f3[BS][H][W], d12a[BS][H][W], d3a[BS][H][W];
double ConcatOut[CHMAX * 2][H][W], BConcatOut[CHMAX * 2][BS][H][W];
double ConcatDelta[CHMAX * 2][H][W], BConcatDelta[CHMAX * 2][BS][H][W];
double CS1delta[CHMAX][H][W], CS2delta[CHMAX][H][W], BCS1delta[CHMAX][BS][H][W], BCS2delta[CHMAX][BS][H][W];
double Gapdelta[CHMAX][H][W], BGapdelta[CHMAX][BS][H][W];
double CaGDelta[CHMAX][H][W], SaGDelta[CHMAX][H][W], BCaGDelta[CHMAX][BS][H][W], BSaGDelta[CHMAX][BS][H][W];
double CunitDelta[CHMAX][H][W], SunitDelta[CHMAX][H][W], BCunitDelta[CHMAX][BS][H][W], BSunitDelta[CHMAX][BS][H][W];

int CBatchNormal = 0; //1で正則化 0でなし

void ReLUActiv(double out[][H][W], int sch, int ech, int hsize, int wsize) {
	int j, h, w;
	for (j = sch; j < ech; j++) //出力側をReLU
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
			if (out[j][h][w] <= 0) out[j][h][w] *= LEAKY;
}
void BReLUActiv(double out[][BS][H][W], int sch, int ech, int hsize, int wsize, int bs) {
	int j, h, w, b;
	for (j = sch; j < ech; j++) for (b = 0; b < bs; b++)//出力側をReLU
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
			if (out[j][b][h][w] <= 0) out[j][b][h][w] *= LEAKY;
}
void ReLUBack(double delta[][H][W], double out[][H][W], int sch, int ech, int hsize, int wsize) {
	int j, h, w;
	for (j = sch; j < ech; j++) //出力側をReLU
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
			if (out[j][h][w] <= 0) delta[j][h][w] *= LEAKY;
}
void BReLUBack(double delta[][BS][H][W], double out[][BS][H][W], int sch, int ech, int hsize, int wsize, int bs) {
	int j, h, w, b;
	for (j = sch; j < ech; j++) for (b = 0; b < bs; b++)//出力側をReLU
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
			if (out[j][b][h][w] <= 0) delta[j][b][h][w] *= LEAKY;
}

void CBatchNormalF(double bnet[][H][W], double anet[][H][W], double* mean, double* var, double gamma, double beta, int hsize, int wsize) {
	double ave, sum, va, sqvar, m = BS * hsize * wsize;
	int b, h, w;
	for (sum = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		sum += bnet[b][h][w];
	ave = sum / m;
	for (sum = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		sum += (ave - bnet[b][h][w]) * (ave - bnet[b][h][w]);
	va = sum / m;
	for (sqvar = sqrt(va + EPS), b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		anet[b][h][w] = gamma * (bnet[b][h][w] - ave) / sqvar + beta;
	*mean = ave;
	*var = va;
}
void CBatchNormalB(double delta[][H][W], double bnet[][H][W], double mean, double var, double* gamma, double* beta, double lr, int hsize, int wsize) {
	int b, h, w;
	double f7 = 1.0 / sqrt(var + EPS), tave, sum, db, dg, ga = *gamma, m = BS * hsize * wsize;

	for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		f3[b][h][w] = (bnet[b][h][w] - mean); //f3 xi-μ
	for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		d12a[b][h][w] = delta[b][h][w] * ga; //d12a
	for (tave = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		tave += f3[b][h][w] * d12a[b][h][w];
	//tave /= m;
	tave /= (var + EPS) * m;
	for (sum = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
		//d3a[b][h][w] = f7 * (d12a[b][h][w] - f3[b][h][w] / (var + EPS) * tave); //d3a
		d3a[b][h][w] = f7 * (d12a[b][h][w] - f3[b][h][w] * tave); //d3a
		sum += d3a[b][h][w];
	}
	sum /= m;
	for (db = 0, dg = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
		db += delta[b][h][w];
		dg += f3[b][h][w] * f7 * delta[b][h][w];
		delta[b][h][w] = d3a[b][h][w] - sum; //dx Delta (doutが更新される)
	}
	*gamma += (-lr * dg / m);
	*beta += (-lr * db / m);
}

void InConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][H][W], double out[][H][W], int ksize,
	double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, oh = H - ksize + 1, ow = W - ksize + 1;
	double reg;
	// convolution
	for (j = 0; j < och; j++) {//出力側
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			for (out[j][h][w] = bias[j], i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
					out[j][h][w] += woi[j][i][y][x] * in[i][h + y][w + x];
		//BNするならここ
		if (BATCHLEARN && CBatchNormal)
			for (reg = gamma[j] / sqrt(mvar[j] + EPS), h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				//(sum - mean[l + 1][j]) / sqrt(var[l + 1][j] + EPS) * gamma[l + 1][j] + beta[l + 1][j]
				out[j][h][w] = (out[j][h][w] - mmean[j]) * reg + beta[j];
	}
	//ReLU
	ReLUActiv(out, 0, och, oh, ow);
}
void BInConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][BS][H][W], double out[][BS][H][W], int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, oh = H - ksize + 1, ow = W - ksize + 1, b;
	// convolution
	for (j = 0; j < och; j++) {//出力側
		if (CBatchNormal) {
			for (b = 0; b < BS; b++)
				for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
					for (bnet[j][b][h][w] = bias[j], i = 0; i < ich; i++) //入力側
						for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
							bnet[j][b][h][w] += woi[j][i][y][x] * in[i][b][h + y][w + x];
			CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], oh, ow);
			mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];
			mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];
		}
		else
			for (b = 0; b < BS; b++)
				for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
					for (out[j][b][h][w] = bias[j], i = 0; i < ich; i++) //入力側
						for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
							out[j][b][h][w] += woi[j][i][y][x] * in[i][b][h + y][w + x];
	}

	//ReLU
	BReLUActiv(out, 0, och, oh, ow, BS);
}

void Conv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, oh = hsize / st, ow = wsize / st;
	double sum, reg;

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// データ入力

	// convolution
	for (j = 0; j < och; j++) {// 出力ノード数
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {// 画像サイズ(高さ)(横幅)
			for (sum = bias[j], i = 0; i < ich; i++) {// 入力ノード数
				for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// カーネルサイズ 
					sum += Pin[i][h + y][w + x] * woi[j][i][y][x];// 畳み込み
			}
			out[j][h / st][w / st] = sum;
		}
		if (BATCHLEARN && CBatchNormal) {// バッチ正規化
			for (reg = gamma[j] / sqrt(mvar[j] + EPS), h = 0; h < oh; h++)for (w = 0; w < ow; w++)// 画像サイズ(高さ)(横幅)
				out[j][h][w] = (out[j][h][w] - mmean[j]) * reg + beta[j];
		}
	}

	// ReLU
	if (rel) ReLUActiv(out, 0, och, oh, ow);
}
void BConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, oh = hsize / st, ow = wsize / st, b;
	double sum;
	// 0 padding
	ph = hsize + ksize - 1; pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];
	// convolution
	for (j = 0; j < och; j++) {//出力側
		if (CBatchNormal) {
			for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {// 画像サイズ			
					for (sum = bias[j], i = 0; i < ich; i++) // 入力channel数
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) // カーネルサイズ 
							sum += BPin[i][b][h + y][w + x] * woi[j][i][y][x];// 畳み込み
					bnet[j][b][h / st][w / st] = sum;
				}
			CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], oh, ow);
			mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];
			mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];
		}
		else {
			for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {// 画像サイズ			
					for (sum = bias[j], i = 0; i < ich; i++) // 入力channel数
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) // カーネルサイズ 
							sum += BPin[i][b][h + y][w + x] * woi[j][i][y][x];// 畳み込み
					out[j][b][h / st][w / st] = sum;
				}
		}
	}
	//出力側をReLU
	if (rel) BReLUActiv(out, 0, och, oh, ow, BS);
}
void GhostConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, oh = hsize / st, ow = wsize / st;
	int gch = och / S;
	double sum, dw_sum, reg;
	// 0 padding
	ph = hsize + ksize - 1; pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// データ入力

	// convolution
	for (j = 0; j < gch; j++) {// 出力 channel数　gch = och / Sまで
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {// 画像サイズ(横幅)			
			for (sum = bias[j], i = 0; i < ich; i++) {// 入力ノード数
				for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// カーネルサイズ 
					sum += Pin[i][h + y][w + x] * woi[j][i][y][x];// 畳み込み
			}
			out[j][h / st][w / st] = sum;
		}
		if (BATCHLEARN && CBatchNormal) // バッチ正規化
			for (reg = gamma[j] / sqrt(mvar[j] + EPS), h = 0; h < oh; h++) for (w = 0; w < ow; w++)// 画像サイズ
				out[j][h][w] = (out[j][h][w] - mmean[j]) * reg + beta[j];
	}
	if (rel) ReLUActiv(out, 0, gch, oh, ow);// ReLU

	ph = oh + ksize - 1; pw = ow + ksize - 1;
	for (i = 0; i < gch; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 0-padding
	for (i = 0; i < gch; i++)for (h = 0; h < oh; h++)for (w = 0; w < ow; w++) Pin[i][h + p][w + p] = out[i][h][w];	// データ入力

	for (i = 0, j = gch; j < och; i++, j++) {//gch以降　線形変換処理(DepthWise畳み込み) 
		if (i >= gch) i = 0; //入力を使いまわす
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++) { // 画像に対するループ (ストライドあり)
			dw_sum = bias[j]; // 畳み込みの合計値を初期化
			for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {// 各入力チャネルに対して畳み込み演算を実行				
				dw_sum += Pin[i][h + y][w + x] * woi[j][0][y][x]; // 各チャネルごとにカーネルを適用
			}
			out[j][h][w] = dw_sum;// 結果を出力に格納
		}
		if (BATCHLEARN && CBatchNormal) // バッチ正規化
			for (reg = gamma[j] / sqrt(mvar[j] + EPS), h = 0; h < oh; h++) for (w = 0; w < ow; w++)// 画像サイズ
				out[j][h][w] = (out[j][h][w] - mmean[j]) * reg + beta[j];
	}

	// ReLU
	if (rel) ReLUActiv(out, gch, och, oh, ow);// gch以降
}
void BGhostConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, oh = hsize / st, ow = wsize / st, b;
	int gch = och / S;
	double sum, dw_sum;
	// 0 padding
	ph = hsize + ksize - 1; pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;							// 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];	// データ入力

	// convolution
	for (j = 0; j < gch; j++) {// 出力 channel数 gch =och / Sまで
		if (CBatchNormal) {
			for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) { // 画像サイズ		
					for (sum = bias[j], i = 0; i < ich; i++) // 入力
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// カーネルサイズ 
							sum += BPin[i][b][h + y][w + x] * woi[j][i][y][x];// 畳み込み
					bnet[j][b][h / st][w / st] = sum;
				}
			//BNするならここ
			CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], oh, ow);
			mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];
			mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];
		}
		else
			for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) { // 画像サイズ	
					for (sum = bias[j], i = 0; i < ich; i++) // 入力
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// カーネルサイズ 
							sum += BPin[i][b][h + y][w + x] * woi[j][i][y][x];// 畳み込み
					out[j][b][h / st][w / st] = sum;
				}
	}

	if (rel) BReLUActiv(out, 0, gch, oh, ow, BS);

	ph = oh + ksize - 1; pw = ow + ksize - 1;
	for (i = 0; i < gch; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;							// 0-padding
	for (i = 0; i < gch; i++)for (b = 0; b < BS; b++)for (h = 0; h < oh; h++)for (w = 0; w < ow; w++) BPin[i][b][h + p][w + p] = out[i][b][h][w];	// データ入力
	//線形変換処理(DepthWise畳み込み)
	for (i = 0, j = gch; j < och; i++, j++) {//gch以降 Depthwise畳み込み処理
		if (i >= gch)i = 0;//入力を使いまわす
		if (CBatchNormal) {
			for (b = 0; b < BS; b++) {
				for (h = 0; h < oh; h++) for (w = 0; w < ow; w++) { // 画像の大きさに対するループ
					dw_sum = bias[j]; // 畳み込みの合計値を初期化					
					for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)
						// 各入力チャネルに対して畳み込み演算を実行
						dw_sum += BPin[i][b][h + y][w + x] * woi[j][0][y][x]; // 各チャネルごとにカーネルを適用				
					bnet[j][b][h][w] = dw_sum;// 結果を出力に格納
				}
			}
			//BNするならここ
			CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], oh, ow);
			mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];
			mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];
		}
		else
			for (b = 0; b < BS; b++) {
				for (h = 0; h < oh; h++) for (w = 0; w < ow; w++) { // 画像ループ
					dw_sum = bias[j]; // 畳み込みの合計値を初期化					
					for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)
						// 各入力チャネルに対して畳み込み演算を実行
						dw_sum += BPin[i][b][h + y][w + x] * woi[j][0][y][x]; // 各チャネルごとにカーネルを適用				
					out[j][b][h][w] = dw_sum;// 結果を出力に格納
				}
			}
	}

	if (rel) BReLUActiv(out, gch, och, oh, ow, BS);//gch以降
}

void CForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double mout[],
	double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX]) {
	int j, s, c, hsize = H, wsize = W, st = 1, n, h, w, y, x;
	double max;
	for (s = 0; s < SEG; s++) {
		//CONV回畳込み（+ReLU)
		for (c = 0; c < CONV; c++) {
			Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, gamma[s][c], beta[s][c], mmean[s][c], mvar[s][c], 1);
			hsize /= st; wsize /= st;
		}
		//PS*PS max pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS)for (w = 0; w < wsize; w += PS, n++) {
					for (max = out[s][c][j][h][w], y = 0; y < PS; y++)for (x = 0; x < PS; x++)//ここにｙとｘのループ
						if (max < out[s][c][j][h + y][w + x])max = out[s][c][j][h + y][w + x];
					mout[n] = max;
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			for (j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS)for (w = 0; w < wsize; w += PS) {
					for (max = out[s][c][j][h][w], y = 0; y < PS; y++)for (x = 0; x < PS; x++)//ここにｙとｘのループ
						if (max < out[s][c][j][h + y][w + x])max = out[s][c][j][h + y][w + x];
					out[s + 1][0][j][h / PS][w / PS] = max;
				}
			}
		}
		hsize /= PS;
		wsize /= PS;
	}
}
void BCForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double mout[][NMAX],
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX]) {
	int j, s, c, hsize = H, wsize = W, st = 1, n, h, w, y, x, b;
	double max;
	for (s = 0; s < SEG; s++) {
		//CONV回畳込み（+ReLU)
		for (c = 0; c < CONV; c++) {
			//for (b = 0; b < BS; b++)
			//	Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c][b], out[s][c + 1][b], hsize, wsize, st, KS);
			BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, bnet[s][c],
				mean[s][c], var[s][c], gamma[s][c], beta[s][c], mmean[s][c], mvar[s][c], 1);
			hsize /= st; wsize /= st;
		}
		//PS*PS max pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (b = 0; b < BS; b++)
				for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
					for (h = 0; h < hsize; h += PS)for (w = 0; w < wsize; w += PS, n++) {
						for (max = out[s][c][j][b][h][w], y = 0; y < PS; y++)for (x = 0; x < PS; x++)//ここにｙとｘのループ
							if (max < out[s][c][j][b][h + y][w + x])max = out[s][c][j][b][h + y][w + x];
						mout[b][n] = max;
					}
				}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			for (j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++) {//出力側
				for (h = 0; h < hsize; h += PS)for (w = 0; w < wsize; w += PS) {
					for (max = out[s][c][j][b][h][w], y = 0; y < PS; y++)for (x = 0; x < PS; x++)//ここにｙとｘのループ
						if (max < out[s][c][j][b][h + y][w + x])max = out[s][c][j][b][h + y][w + x];
					out[s + 1][0][j][b][h / PS][w / PS] = max;
				}
			}
		}
		hsize /= PS;
		wsize /= PS;
	}
}

void SkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double mout[],
	double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double max;
	//Input
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding なしの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], 1);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}
		for (n = 0; n < chN[s][c]; n++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			skipOut[s][n][h][w] = out[s][c][n][h][w] + out[s][0][n][h][w];// Skip-Connection
		//本来のResNetではここでReLUを行う

		if (s == SEG - 1) {// Max Pooling// out[s][c] -> mout 全結合に入れる 
			for (fn = 0, n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS)for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++)for (j = 0; j < PS; j++)// max計算
						if (max < skipOut[s][n][h + i][w + j]) max = skipOut[s][n][h + i][w + j];
					mout[fn] = max;// 全結合の入力
					fn++;
				}
			}
		}
		else {// 次のセグメントの0層にmax代入 stride=PS 
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0], 1);
			for (n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						max += skipOut[s][n][h + i][w + j];
					out[s + 1][0][n][h / PS][w / PS] += max / (PS * PS);// Skip-Connection
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}
void BSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double mout[][NMAX],
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1, b;// st:ストライドサイズ
	double max;
	//Input
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding なしの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], 1);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}
		for (n = 0; n < chN[s][c]; n++)for (b = 0; b < BS; b++)
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][b][h][w] = out[s][c][n][b][h][w] + out[s][0][n][b][h][w];// Skip-Connection
		//本来のResNetではここでReLUを行う

		if (s == SEG - 1) {// Max Pooling// out[s][c] -> mout 全結合に入れる 
			for (b = 0; b < BS; b++)
				for (fn = 0, n = 0; n < chN[s][c]; n++) {
					for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
						for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)// max計算
							if (max < skipOut[s][n][b][h + i][w + j])
								max = skipOut[s][n][b][h + i][w + j];
						mout[b][fn] = max;// 全結合の入力
						fn++;
					}
				}
		}
		else {// 次のセグメントの0層にmax代入
			BConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0], 1);
			for (n = 0; n < chN[s][c]; n++) for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						max += skipOut[s][n][b][h + i][w + j];
					out[s + 1][0][n][b][h / PS][w / PS] += max / (PS * PS);// Skip-Connection
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}

//Ghostバッチ化後にやる→設計中
void GhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double ggamma[], double gbeta[], double gmmean[], double gmvar[]) {
	int j, h, w, oh = hsize / st, ow = wsize / st;
	//1回目GhostModule→出力をginへ格納
	GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);

	//2回目GhostModule→出力をgoutへ格納
	GhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);

	//スキップ接続(och/S)
	for (j = 0; j < ich; j++)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[j][h][w] = gout[j][h][w] + in[j][h * st][w * st];// Skip-Connection
	for (; j < och; j++) // och> ichの場合はSkip 0
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[j][h][w] = gout[j][h][w];
}
void BGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], //double gnet[][BS][H][W],
	double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double gmmean[], double gmvar[]) {
	int i, h, w, oh = hsize / st, ow = wsize / st, b;
	//1回目GhostModule→出力をginへ格納
	BGhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, bnet, mean, var, gamma, beta, mmean, mvar, 1);
	//2回目GhostModule→出力をgoutへ格納
	BGhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, gbnet, gmean, gvar, ggamma, gbeta, gmmean, gmvar, 0);

	//スキップ接続(～ich)
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[i][b][h][w] = gout[i][b][h][w] + in[i][b][h * st][w * st];// Skip-Connection
	for (; i < och; i++) for (b = 0; b < BS; b++) // och> ichの場合はSkip 0		
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[i][b][h][w] = gout[i][b][h][w];
}

void GForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double mout[],
	double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][H][W], double gout[SEG][CONV][CHMAX][H][W],
	double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double max;
	//Input
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding なしの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			//GhostCONV→完了後(済)→GB(今)
			if (s == 0) Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], 1);
			//else GhostConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],1);
			else
				GhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
					gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],
					gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c]);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}

		//スキップ接続 
		if (s == 0)
			for (n = 0; n < chN[s][c]; n++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][h][w] = out[s][c][n][h][w] + out[s][0][n][h][w];// Skip-Connection
		else
			for (n = 0; n < chN[s][c]; n++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][h][w] = out[s][c][n][h][w];//no Skip-Connection

			// Max Pooling
		if (s == SEG - 1) {// out[s][c] -> mout 全結合に入れる 
			for (fn = 0, n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					// max計算
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						if (max < skipOut[s][n][h + i][w + j])
							max = skipOut[s][n][h + i][w + j];
					// 全結合の入力
					mout[fn] = max;
					fn++;
				}
			}
		}
		else {// 次のセグメントの0層にmax代入
			//本当はCONVだけどGhostCONV→完了後→GhostBottle
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0], 1);
			//GhostConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);
			for (n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						max += skipOut[s][n][h + i][w + j];
					out[s + 1][0][n][h / PS][w / PS] += max / (PS * PS);// Skip-Connection
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}
void BGForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double mout[][NMAX],
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], //
	double gbnet[SEG][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1, b;// st:ストライドサイズ
	double max;
	//Input
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding なしの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			if (s == 0) BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], 1);
			//else BGhostConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
			//	bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],1);
			else
				BGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
					bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],
					gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c]);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}
		if (s == 0)
			for (n = 0; n < chN[s][c]; n++)for (b = 0; b < BS; b++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][b][h][w] = out[s][c][n][b][h][w] + out[s][0][n][b][h][w];// Skip-Connection
		else
			for (n = 0; n < chN[s][c]; n++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][b][h][w] = out[s][c][n][b][h][w];//no Skip-Connection

		//本来のResNetではここでReLUを行う

		// Max Pooling
		if (s == SEG - 1) {// out[s][c] -> mout 全結合に入れる 
			for (b = 0; b < BS; b++)
				for (fn = 0, n = 0; n < chN[s][c]; n++) {
					for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
						// max計算
						for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
							if (max < skipOut[s][n][b][h + i][w + j])
								max = skipOut[s][n][b][h + i][w + j];
						// 全結合の入力
						mout[b][fn] = max;
						fn++;
					}
				}
		}
		else {// 次のセグメントの0層にmax代入
			BConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0], 1);
			for (n = 0; n < chN[s][c]; n++) for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						max += skipOut[s][n][b][h + i][w + j];
					out[s + 1][0][n][b][h / PS][w / PS] += max / (PS * PS);// Skip-Connection
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}

void PConv(int ich, int och, double bias[], double woi[][CHMAX], double in[][H][W], double out[][H][W], int hsize, int wsize,
	double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, j, h, w;
	double sum, reg;

	// 1x1 Convolution
	for (j = 0; j < och; j++) { // 出力ノード数
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) { // 画像サイズ(高さ)(横幅)
			for (sum = bias[j], i = 0; i < ich; i++) // バイアスを初期化 // 入力ノード数					
				sum += in[i][h][w] * woi[j][i];// 1x1の畳み込み
			out[j][h][w] = sum;
		}
		//if (BATCHLEARN && CBatchNormal) {// バッチ正規化
		//	for (reg= gamma[j] / sqrt(mvar[j] + EPS), h = 0; h < hsize; h++)  // 画像サイズ(高さ)
		//		for (w = 0; w < wsize; w++)  // 画像サイズ(横幅)
		//			out[j][h][w] = (out[j][h][w] - mmean[j]) * reg + beta[j];
		//}
	}

	if (rel) ReLUActiv(out, 0, och, hsize, wsize);
}
void BPConv(int ich, int och, double bias[], double woi[][CHMAX], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize,
	double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, b, j, h, w;
	double sum;

	// 1x1 Convolution
	for (j = 0; j < och; j++) { // 出力ノード数
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) { // 画像サイズ(高さ)(横幅)
				for (sum = bias[j], i = 0; i < ich; i++)  // バイアスを初期化// 入力ノード数						
					sum += in[i][b][h][w] * woi[j][i];// 1x1の畳み込み
				out[j][b][h][w] = sum;
			}
		}
		//if (CBatchNormal) {// バッチ正規化
		//	//BNするならここ
		//	CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], hsize, wsize);
		//	mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];
		//	mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];
		//}
	}

	// ReLU
	if (rel) BReLUActiv(out, 0, och, hsize, wsize, BS);
}
void CatPConv(int ich, int och, double bias[], double woi[][CHMAX * 2], double in[][H][W], double out[][H][W], int hsize, int wsize,
	double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, j, h, w;
	double sum, reg;

	// 1x1 Convolution
	for (j = 0; j < och; j++) { // 出力ノード数
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) { // 画像サイズ(高さ)(横幅)
			for (sum = bias[j], i = 0; i < ich; i++) // バイアスを初期化 // 入力ノード数					
				sum += in[i][h][w] * woi[j][i];// 1x1の畳み込み
			out[j][h][w] = sum;
		}
		//if (BATCHLEARN && CBatchNormal) {// バッチ正規化
		//	for (reg= gamma[j] / sqrt(mvar[j] + EPS), h = 0; h < hsize; h++)  // 画像サイズ(高さ)
		//		for (w = 0; w < wsize; w++)  // 画像サイズ(横幅)
		//			out[j][h][w] = (out[j][h][w] - mmean[j]) * reg + beta[j];
		//}
	}

	// ReLU
	if (rel) ReLUActiv(out, 0, och, hsize, wsize);
}
void BCatPConv(int ich, int och, double bias[], double woi[][CHMAX * 2], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize,
	double gamma[], double beta[], double mmean[], double mvar[], int rel) {
	int i, b, j, h, w;
	double sum;

	// 1x1 Convolution
	for (j = 0; j < och; j++) { // 出力ノード数
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) { // 画像サイズ(高さ)(横幅)
				for (sum = bias[j], i = 0; i < ich; i++) // バイアスを初期化 // 入力ノード数					
					sum += in[i][b][h][w] * woi[j][i];// 1x1の畳み込み
				out[j][b][h][w] = sum;
			}
		}
		//if (CBatchNormal) {// バッチ正規化
		//	CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], hsize, wsize);
		//	mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];
		//	mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];
		//}
	}

	// ReLU
	if (rel) BReLUActiv(out, 0, och, hsize, wsize, BS);
}

void CAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double in[][H][W], double out[][H][W], int hsize, int wsize,
	double gamma[][CHMAX], double beta[][CHMAX], double mmean[][CHMAX], double mvar[][CHMAX], double cain[][H][W], double caout[][H][W], double gapin[][H][W]) {
	int i, h, w;
	double gap_sum;

	//CAUNit(Pooling→CONV(ReLU込)→CONV→シグモイド→C*1*1圧縮→処理前入力*処理後の積→出力

	// Global Average Pooling
	for (i = 0; i < ich; i++) {
		// 各チャンネルの合計値を計算
		for (gap_sum = 0.0, h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) gap_sum += in[i][h][w];
		gapin[i][0][0] = gap_sum / (hsize * wsize);// 平均値を計算し、1x1の出力に格納
	}

	//1回目CONV→出力をcainへ格納
	PConv(ich, ich, bias[0], woi[0], gapin, cain, 1, 1, gamma[0], beta[0], mmean[0], mvar[0], 1);
	//2回目CONV→出力をgoutへ格納
	PConv(ich, ich, bias[1], woi[1], cain, caout, 1, 1, gamma[1], beta[1], mmean[1], mvar[1], 0);
	//シグモイド
	for (i = 0; i < ich; i++) caout[i][0][0] = 1.0 / (1.0 + exp(-caout[i][0][0])); //sigmoid

	//処理前と処理後の積計算
	for (i = 0; i < ich; i++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		out[i][h][w] = in[i][h][w] * caout[i][0][0];// 処理前と処理後の積計算
}
void BCAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize,
	double gamma[][CHMAX], double beta[][CHMAX], double mmean[][CHMAX], double mvar[][CHMAX], double cain[][BS][H][W], double caout[][BS][H][W], double gapin[][BS][H][W]) {
	int i, h, w, b;
	double gap_sum, div=(double)(hsize*wsize);

	//CAUNit(Pooling→CONV(ReLU込)→CONV→シグモイド→C*1*1圧縮→処理前入力*処理後の積→出力

	// Global Average Pooling
	for (i = 0; i < ich; i++) {
		for (b = 0; b < BS; b++) {
			gap_sum = 0.0; // 各チャンネルの合計値を初期化
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) gap_sum += in[i][b][h][w];
			gapin[i][b][0][0] = gap_sum / div;// 平均値を計算し、1x1の出力に格納
		}
	}

	//1回目CONV→出力をcainへ格納
	BPConv(ich, ich, bias[0], woi[0], gapin, cain, 1, 1, gamma[0], beta[0], mmean[0], mvar[0], 1);
	//2回目CONV→出力をgoutへ格納
	BPConv(ich, ich, bias[1], woi[1], cain, caout, 1, 1, gamma[1], beta[1], mmean[1], mvar[1], 0);
	//シグモイド
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++)
		caout[i][b][0][0] = 1.0 / (1.0 + exp(-caout[i][b][0][0])); //sigmoid

	//処理前と処理後の積計算
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++) {
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			out[i][b][h][w] = in[i][b][h][w] * caout[i][b][0][0];// 処理前と処理後の積計算
	}
}

void SAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double in[][H][W], double out[][H][W], int hsize, int wsize,
	double gamma[][CHMAX], double beta[][CHMAX], double mmean[][CHMAX], double mvar[][CHMAX], double sain[][H][W], double saout[][H][W]) {
	int i, h, w;

	//SAUNit(CONV(ReLU込)→CONV→シグモイド→1*H*W→処理前入力*処理後の積→出力

	//1回目CONV→出力をsainへ格納
	PConv(ich, ich, bias[0], woi[0], in, sain, hsize, wsize, gamma[0], beta[0], mmean[0], mvar[0], 1);
	//2回目CONV→出力をsaoutへ格納
	PConv(ich, 1, bias[1], woi[1], sain, saout, hsize, wsize, gamma[1], beta[1], mmean[1], mvar[1], 0);
	//シグモイド
	for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		saout[0][h][w] = 1.0 / (1.0 + exp(-saout[0][h][w])); //sigmoid

	//処理前と処理後の積計算
	for (i = 0; i < ich; i++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		out[i][h][w] = in[i][h][w] * saout[0][h][w];// 処理前と処理後の積計算
}
void BSAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize,
	double gamma[][CHMAX], double beta[][CHMAX], double mmean[][CHMAX], double mvar[][CHMAX], double sain[][BS][H][W], double saout[][BS][H][W]) {
	int i, h, w, b;

	//SAUNit(CONV(ReLU込)→CONV→シグモイド→1*H*W→処理前入力*処理後の積→出力

	//1回目CONV→出力をsainへ格納
	BPConv(ich, ich, bias[0], woi[0], in, sain, hsize, wsize, gamma[0], beta[0], mmean[0], mvar[0], 1);
	//2回目CONV→出力をsaoutへ格納
	BPConv(ich, 1, bias[1], woi[1], sain, saout, hsize, wsize, gamma[1], beta[1], mmean[1], mvar[1], 0);
	//シグモイド
	for (b = 0; b < BS; b++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		saout[0][b][h][w] = 1.0 / (1.0 + exp(-saout[0][b][h][w])); //sigmoid

	//処理前と処理後の積計算
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++) {
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			out[i][b][h][w] = in[i][b][h][w] * saout[0][b][h][w];// 処理前と処理後の積計算
	}
}

void CSGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double ggamma[], double gbeta[], double gmmean[], double gmvar[],
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][H][W], double cagamma[][CHMAX], double cabeta[][CHMAX], double cammean[][CHMAX], double camvar[][CHMAX], double cunitout[][H][W], double gapin[][H][W],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][H][W], double sagamma[][CHMAX], double sabeta[][CHMAX], double sammean[][CHMAX], double samvar[][CHMAX], double sunitout[][H][W],
	double resbias[], double reswoi[][CHMAX * 2], double resout[][H][W]) {
	int j, h, w, oh = hsize / st, ow = wsize / st;

	//1回目GhostModule→出力をginへ格納
	GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);

	//2回目GhostModule→出力をgoutへ格納
	GhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);

	////提案手法(CAUnit+SAUnit統合)
	////CAUNit(Global Average Pooling→PConv(ReLU込)→PConv→シグモイド→C*1*1圧縮→処理前入力*処理後の積→出力
	CAUnit(och, cabias, cawoi, gout, cunitout, oh, ow, cagamma, cabeta, cammean, camvar, caout[0], caout[1], gapin);
	////SAUNit(PConv(ReLU込)→PConv→シグモイド→1*h*w圧縮→処理前入力*処理後の積→出力
	SAUnit(och / S, sabias, sawoi, gout, sunitout, oh, ow, sagamma, sabeta, sammean, samvar, saout[0], saout[1]);

	/*SAUnit(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
		gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],
		sabias[s][c], sawoi[s][c], sain[s][c], saout[s][c]);*/
		////concat
	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			ConcatOut[j][h][w] = cunitout[j][h][w];
	}
	for (; j < och + och / S; j++) {
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			ConcatOut[j][h][w] = sunitout[j - och][h][w];
	}
	////PConv
	CatPConv(och + och / S, och, resbias, reswoi, ConcatOut, resout, oh, ow, gamma, beta, mmean, mvar, 0);

	//スキップ接続
	for (j = 0; j < ich; j++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][h][w] = resout[j][h][w] + in[j][h * st][w * st];
	for (; j < och; j++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][h][w] = resout[j][h][w];
}
void BCSGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double gmmean[], double gmvar[],
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][BS][H][W], double cagamma[][CHMAX], double cabeta[][CHMAX], double cammean[][CHMAX], double camvar[][CHMAX], double cunitout[][BS][H][W], double gapin[][BS][H][W],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][BS][H][W], double sagamma[][CHMAX], double sabeta[][CHMAX], double sammean[][CHMAX], double samvar[][CHMAX], double sunitout[][BS][H][W],
	double resbias[], double reswoi[][CHMAX * 2], double resout[][BS][H][W]) {
	int j, h, w, oh = hsize / st, ow = wsize / st, b;

	//1回目GhostModule→出力をginへ格納
	BGhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, bnet, mean, var, gamma, beta, mmean, mvar, 1);
	//2回目GhostModule→出力をgoutへ格納
	BGhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, gbnet, gmean, gvar, ggamma, gbeta, gmmean, gmvar, 0);
	////提案手法(CAUnit+SAUnit統合)
	////CAUNit(Global Average Pooling→PConv(ReLU込)→PConv→シグモイド→C*1*1圧縮→処理前入力*処理後の積→出力
	BCAUnit(och, cabias, cawoi, gout, cunitout, oh, ow, cagamma, cabeta, cammean, camvar, caout[0], caout[1], gapin);
	////SAUNit(PConv(ReLU込)→PConv→シグモイド→1*h*w圧縮→処理前入力*処理後の積→出力
	BSAUnit(och / S, sabias, sawoi, gout, sunitout, oh, ow, sagamma, sabeta, sammean, samvar, saout[0], saout[1]);

	/*SAUnit(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
		gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],
		sabias[s][c], sawoi[s][c], sain[s][c], saout[s][c]);*/
		////concat
	for (j = 0; j < och; j++) for (b = 0; b < BS; b++)//(要確認)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			BConcatOut[j][b][h][w] = cunitout[j][b][h][w];
	for (; j < och + och / S; j++) for (b = 0; b < BS; b++)//(要確認)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			BConcatOut[j][b][h][w] = sunitout[j - och][b][h][w];

	////PConv
	BCatPConv(och + och / S, och, resbias, reswoi, BConcatOut, resout, oh, ow, gamma, beta, mmean, mvar, 0);

	//スキップ接続
	for (j = 0; j < ich; j++) for (b = 0; b < BS; b++)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[j][b][h][w] = resout[j][b][h][w] + in[j][b][h * st][w * st];
	for (; j < och; j++) for (b = 0; b < BS; b++)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[j][b][h][w] = resout[j][b][h][w];
}

void CGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[], double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double ggamma[], double gbeta[], double gmmean[], double gmvar[],
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][H][W], double cagamma[][CHMAX], double cabeta[][CHMAX], double cammean[][CHMAX], double camvar[][CHMAX], double cunitout[][H][W], double gapin[][H][W]) {
	int j, h, w, oh = hsize / st, ow = wsize / st;

	//1回目GhostModule→出力をginへ格納
	GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);

	//2回目GhostModule→出力をgoutへ格納
	GhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);

	////提案手法(CAUnit+SAUnit統合)
	////CAUNit(Global Average Pooling→PConv(ReLU込)→PConv→シグモイド→C*1*1圧縮→処理前入力*処理後の積→出力
	CAUnit(och, cabias, cawoi, gout, cunitout, oh, ow, cagamma, cabeta, cammean, camvar, caout[0], caout[1], gapin);

	//スキップ接続
	for (j = 0; j < ich; j++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][h][w] = cunitout[j][h][w] + in[j][h * st][w * st];
	for (; j < och; j++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][h][w] = cunitout[j][h][w];
}
void BCGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double gmmean[], double gmvar[],
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][BS][H][W], double cagamma[][CHMAX], double cabeta[][CHMAX], double cammean[][CHMAX], double camvar[][CHMAX], double cunitout[][BS][H][W], double gapin[][BS][H][W]) {
	int j, h, w, oh = hsize / st, ow = wsize / st, b;

	//1回目GhostModule→出力をginへ格納
	BGhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, bnet, mean, var, gamma, beta, mmean, mvar, 1);
	//2回目GhostModule→出力をgoutへ格納
	BGhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, gbnet, gmean, gvar, ggamma, gbeta, gmmean, gmvar, 0);
	////提案手法(CAUnit+SAUnit統合)
	////CAUNit(Global Average Pooling→PConv(ReLU込)→PConv→シグモイド→C*1*1圧縮→処理前入力*処理後の積→出力
	BCAUnit(och, cabias, cawoi, gout, cunitout, oh, ow, cagamma, cabeta, cammean, camvar, caout[0], caout[1], gapin);

	//スキップ接続
	for (j = 0; j < ich; j++) for (b = 0; b < BS; b++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][b][h][w] = cunitout[j][b][h][w] + in[j][b][h * st][w * st];
	for (; j < och; j++) for (b = 0; b < BS; b++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][b][h][w] = cunitout[j][b][h][w];
}

void SGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[], double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double ggamma[], double gbeta[], double gmmean[], double gmvar[],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][H][W], double sagamma[][CHMAX], double sabeta[][CHMAX], double sammean[][CHMAX], double samvar[][CHMAX], double sunitout[][H][W]) {
	int j, h, w, oh = hsize / st, ow = wsize / st;

	//1回目GhostModule→出力をginへ格納
	GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);

	//2回目GhostModule→出力をgoutへ格納
	GhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);

	////SAUnit only
	SAUnit(och, sabias, sawoi, gout, sunitout, oh, ow, sagamma, sabeta, sammean, samvar, saout[0], saout[1]);

	//スキップ接続
	for (j = 0; j < ich; j++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][h][w] = sunitout[j][h][w] + in[j][h * st][w * st];
	for (; j < och; j++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		out[j][h][w] = sunitout[j][h][w];
}
void BSGhostBottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double gmmean[], double gmvar[],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][BS][H][W], double sagamma[][CHMAX], double sabeta[][CHMAX], double sammean[][CHMAX], double samvar[][CHMAX], double sunitout[][BS][H][W]) {
	int j, h, w, oh = hsize / st, ow = wsize / st, b;

	//1回目GhostModule→出力をginへ格納
	BGhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, bnet, mean, var, gamma, beta, mmean, mvar, 1);
	//2回目GhostModule→出力をgoutへ格納
	BGhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, gbnet, gmean, gvar, ggamma, gbeta, gmmean, gmvar, 0);
	////SAUnit only
	////SAUNit(PConv(ReLU込)→PConv→シグモイド→1*h*w圧縮→処理前入力*処理後の積→出力
	BSAUnit(och, sabias, sawoi, gout, sunitout, oh, ow, sagamma, sabeta, sammean, samvar, saout[0], saout[1]);

	//スキップ接続
	for (j = 0; j < ich; j++) for (b = 0; b < BS; b++)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[j][b][h][w] = sunitout[j][b][h][w] + in[j][b][h * st][w * st];
	for (; j < och; j++) for (b = 0; b < BS; b++)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			out[j][b][h][w] = sunitout[j][b][h][w];
}

void CSGForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][H][W], double gout[SEG][CONV][CHMAX][H][W], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][H][W], double cagamma[][CONV][2][CHMAX], double cabeta[][CONV][2][CHMAX], double cammean[][CONV][2][CHMAX], double camvar[][CONV][2][CHMAX], double cunitout[SEG][CONV][CHMAX][H][W], double gapin[SEG][CONV][CHMAX][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][H][W], double sagamma[][CONV][2][CHMAX], double sabeta[][CONV][2][CHMAX], double sammean[][CONV][2][CHMAX], double samvar[][CONV][2][CHMAX], double sunitout[SEG][CONV][CHMAX][H][W],
	double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][H][W]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double max;
	//Input
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding なしの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			//GhostCONV→完了後(済)→GB(今)
			if (s == 0) Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], 1);
			//else if (s == SEG - 1) GhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],
			//	gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c]);
			/*else CSGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c],
				cabias[s][c], cawoi[s][c], caout[s][c], cagamma[s][c + 1], cabeta[s][c + 1], cammean[s][c + 1], camvar[s][c + 1], cunitout[s][c], gapin[s][c],
				sabias[s][c], sawoi[s][c], saout[s][c], sagamma[s][c + 1], sabeta[s][c + 1], sammean[s][c + 1], samvar[s][c + 1], sunitout[s][c], resbias[s][c], reswoi[s][c], resout[s][c]);*/
			/*else CGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
			gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c],
			cabias[s][c], cawoi[s][c], caout[s][c], cagamma[s][c + 1], cabeta[s][c + 1], cammean[s][c + 1], camvar[s][c + 1], cunitout[s][c], gapin[s][c]);*/
			else SGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c],
				sabias[s][c], sawoi[s][c], saout[s][c], sagamma[s][c + 1], sabeta[s][c + 1], sammean[s][c + 1], samvar[s][c + 1], sunitout[s][c]);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}

		//スキップ接続 
		if (s == 0) for (n = 0; n < chN[s][c]; n++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) {
			skipOut[s][n][h][w] = out[s][c][n][h][w] + out[s][0][n][h][w];// Skip-Connection
		}
		else  for (n = 0; n < chN[s][c]; n++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) {
			skipOut[s][n][h][w] = out[s][c][n][h][w];//no Skip-Connection
		}
		// Max Pooling
		if (s == SEG - 1) {// out[s][c] -> mout 全結合に入れる 
			for (fn = 0, n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					// max計算
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						if (max < skipOut[s][n][h + i][w + j]) max = skipOut[s][n][h + i][w + j];
					// 全結合の入力
					mout[fn] = max;
					fn++;
				}
			}
		}
		else {// 次のセグメントの0層にmax代入
			//本当はCONVだけどGhostCONV→完了後→GhostBottle
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0], 1);
			//GhostConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);
			for (n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						max += skipOut[s][n][h + i][w + j];
					out[s + 1][0][n][h / PS][w / PS] += max / (PS * PS);// Skip-Connection
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}
void BCSGForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], double gbnet[SEG][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][BS][H][W], double cagamma[][CONV][2][CHMAX], double cabeta[][CONV][2][CHMAX], double cammean[][CONV][2][CHMAX], double camvar[][CONV][2][CHMAX], double cunitout[SEG][CONV][CHMAX][BS][H][W], double gapin[SEG][CONV][CHMAX][BS][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][BS][H][W], double sagamma[][CONV][2][CHMAX], double sabeta[][CONV][2][CHMAX], double sammean[][CONV][2][CHMAX], double samvar[][CONV][2][CHMAX], double sunitout[SEG][CONV][CHMAX][BS][H][W],
	double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][BS][H][W]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1, b;// st:ストライドサイズ
	double max;
	//Input
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding なしの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			//GhostCONV→完了後(済)→GB(今)
			if (s == 0) BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], 1);
			//else if (s == SEG - 1) BGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
			//	bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1],
			//	gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c]);
			/*else BCSGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c],
				cabias[s][c], cawoi[s][c], caout[s][c], cagamma[s][c + 1], cabeta[s][c + 1], cammean[s][c + 1], camvar[s][c + 1], cunitout[s][c], gapin[s][c],
				sabias[s][c], sawoi[s][c], saout[s][c], sagamma[s][c + 1], sabeta[s][c + 1], sammean[s][c + 1], samvar[s][c + 1], sunitout[s][c], resbias[s][c], reswoi[s][c], resout[s][c]);*/
			/*else BCGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
			bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c],
			cabias[s][c], cawoi[s][c], caout[s][c], cagamma[s][c + 1], cabeta[s][c + 1], cammean[s][c + 1], camvar[s][c + 1], cunitout[s][c], gapin[s][c]);*/
			else BSGhostBottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], gmmean[s][c], gmvar[s][c],
				sabias[s][c], sawoi[s][c], saout[s][c], sagamma[s][c + 1], sabeta[s][c + 1], sammean[s][c + 1], samvar[s][c + 1], sunitout[s][c]);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}

		//スキップ接続 
		if (s == 0) for (n = 0; n < chN[s][c]; n++)for (b = 0; b < BS; b++)
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][b][h][w] = out[s][c][n][b][h][w] + out[s][0][n][b][h][w];// Skip-Connection
		else for (n = 0; n < chN[s][c]; n++)for (b = 0; b < BS; b++)
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				skipOut[s][n][b][h][w] = out[s][c][n][b][h][w];//no Skip-Connection

		// Max Pooling
		if (s == SEG - 1) {// out[s][c] -> mout 全結合に入れる 
			for (b = 0; b < BS; b++)
				for (fn = 0, n = 0; n < chN[s][c]; n++) {
					for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
						// max計算
						for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
							if (max < skipOut[s][n][b][h + i][w + j])
								max = skipOut[s][n][b][h + i][w + j];
						// 全結合の入力
						mout[b][fn] = max;
						fn++;
					}
				}
		}
		else {// 次のセグメントの0層にmax代入
			//本当はCONVだけどGhostCONV→完了後→GhostBottle
			BConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0], 1);
			//GhostConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);
			for (n = 0; n < chN[s][c]; n++) for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					for (max = 0, i = 0; i < PS; i++) for (j = 0; j < PS; j++)
						max += skipOut[s][n][b][h + i][w + j];
					out[s + 1][0][n][b][h / PS][w / PS] += max / (PS * PS);// Skip-Connection
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}

void InBackConv(int ich, int och, double outdelta[][H][W], int hsize, int wsize, int st, int ksize,
	double in[][H][W], double dbias[], double dwoi[][ICH][IKS][IKS], double out[][H][W]) {
	int i, j, h, w, y, x;
	int oh = (H - ksize + 1) / st, ow = (W - ksize + 1) / st;

	//作られた入力側の誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し)
	ReLUBack(outdelta, out, 0, och, oh, ow);

	// convolution
	for (j = 0; j < och; j++) {//出力側
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			for (dbias[j] += outdelta[j][h][w], i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
					//out[j][h][w] += woi[j][i][y][x] * in[i][h*st + y][w*st + x];
					dwoi[j][i][y][x] += in[i][h + y][w + x] * outdelta[j][h][w];
	}
}
void BInBackConv(int ich, int och, double outdelta[][BS][H][W], int hsize, int wsize, int st, int ksize,
	double in[][BS][H][W], double dbias[], double dwoi[][ICH][IKS][IKS], double out[][BS][H][W],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double lr) {
	int i, j, h, w, y, x, b;
	int oh = (H - ksize + 1) / st, ow = (W - ksize + 1) / st;

	//作られた入力側の誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し)
	BReLUBack(outdelta, out, 0, och, oh, ow, BS);
	if (CBatchNormal) for (j = 0; j < och; j++) //BNするならここ
		CBatchNormalB(outdelta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], lr, oh, ow);

	// convolution
	for (j = 0; j < och; j++) {//出力側
		for (b = 0; b < BS; b++)
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				for (dbias[j] += outdelta[j][b][h][w], i = 0; i < ich; i++) //入力側
					for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
						//out[j][h][w] += woi[j][i][y][x] * in[i][h + y][w + x];
						dwoi[j][i][y][x] += in[i][b][h + y][w + x] * outdelta[j][b][h][w];
	}
}

void BackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][CHMAX][KS][KS], double out[][H][W], int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, oh = hsize / st, ow = wsize / st;
	double delj;
	//誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し)
	if (rel) ReLUBack(delta, out, 0, och, oh, ow);

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pdelta[i][h][w] = 0;						// デルタ 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 入力 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// データ入力

	// back convolution
	for (j = 0; j < och; j++) {//出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = delta[j][h / st][w / st];
			for (dBias[j] += delj, i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++) {//ここにｙとｘのループ
					//前向きで使ったところと同じ関係で誤差信号と重みの修正量を蓄積させる
					//out[j][h][w] += woi[j][i][y][x] * Pin[i][h * st + y][w * st + x];
					Pdelta[i][h + y][w + x] += woi[j][i][y][x] * delj;
					dWoi[j][i][y][x] += Pin[i][h + y][w + x] * delj;
				}
		}
	}

	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		indelta[i][h][w] = Pdelta[i][h + p][w + p];	// デルタのパディング削除
}
void BBackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double outdelta[][BS][H][W],
	int hsize, int wsize, int st, int ksize, double in[][BS][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][BS][H][W],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double lr, int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	int oh = hsize / st, ow = wsize / st, b;
	double delj;
	//誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し)
	if (rel) BReLUBack(outdelta, out, 0, och, oh, ow, BS);
	if (CBatchNormal) for (j = 0; j < och; j++) //BNするならここ
		CBatchNormalB(outdelta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], lr, oh, ow);

	// 0 padding
	ph = hsize + ksize - 1; pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPdelta[i][b][h][w] = 0;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];

	// back convolution
	for (j = 0; j < och; j++) for (b = 0; b < BS; b++) {//出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = outdelta[j][b][h / st][w / st];
			for (dbias[j] += delj, i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++) {//ここにｙとｘのループ
					//前向きで使ったところと同じ関係で誤差信号と重みの修正量を蓄積させる
					//out[j][h][w] += woi[j][i][y][x] * Pin[i][h * st + y][w * st + x];
					BPdelta[i][b][h + y][w + x] += woi[j][i][y][x] * delj;
					dwoi[j][i][y][x] += BPin[i][b][h + y][w + x] * delj;
				}
		}
	}
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++) //パディング戻し
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			indelta[i][b][h][w] = BPdelta[i][b][h + p][w + p];
}

void BackGhostConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][CHMAX][KS][KS], double out[][H][W], int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	int oh = hsize / st, ow = wsize / st, gch = och / S;
	double delj;

	//出力j側の誤差信号に活性化関数の微分を適用(outDeltaのReLU戻し) ghost分
	if (rel) ReLUBack(delta, out, gch, och, oh, ow);

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;

	for (i = 0; i < gch; i++) for (h = 0; h < ph; h++) for (w = 0; w < pw; w++) Pin[i][h][w] = 0; // 0-padding
	for (i = 0; i < gch; i++) for (h = 0; h < ph; h++) for (w = 0; w < pw; w++) Pdelta[i][h][w] = 0; // 0-padding
	for (i = 0; i < gch; i++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++) Pin[i][h + p][w + p] = out[i][h][w];

	//線形変換処理(DepthWise畳み込み)
	for (i = 0, j = gch; j < och; i++, j++) { // ghost チャネルを処理
		if (i >= gch) i = 0;
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++) {
			dBias[j] += delta[j][h][w];
			for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {
				dWoi[j][0][y][x] += Pin[i][h + y][w + x] * delta[j][h][w];
				Pdelta[i][h + y][w + x] += woi[j][0][y][x] * delta[j][h][w];// 勾配計算
			}
		}
	}
	// ghost deltaを上乗せ
	for (i = 0; i < gch; i++) for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
		delta[i][h][w] += Pdelta[i][h + p][w + p]; // デルタのパディング削除

	//出力j側の誤差信号に活性化関数の微分を適用(outDeltaのReLU戻し) 通常分
	if (rel) ReLUBack(delta, out, 0, gch, oh, ow);

	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pdelta[i][h][w] = 0;						// デルタ 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 入力 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// データ入力

	// back convolution
	for (j = 0; j < gch; j++) {//出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = delta[j][h / st][w / st];
			for (dBias[j] += delj, i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++) {//ここにｙとｘのループ
					//前向きで使ったところと同じ関係で誤差信号と重みの修正量を蓄積させる
					//out[j][h][w] += woi[j][i][y][x] * Pin[i][h * st + y][w * st + x];
					Pdelta[i][h + y][w + x] += woi[j][i][y][x] * delj;
					dWoi[j][i][y][x] += Pin[i][h + y][w + x] * delj;
				}
		}
	}

	//作られた入力側の誤差信号
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		indelta[i][h][w] = Pdelta[i][h + p][w + p];	// デルタのパディング削除
}
void BBackGhostConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double outdelta[][BS][H][W],
	int hsize, int wsize, int st, int ksize, double in[][BS][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][BS][H][W],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double lr, int rel) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	int oh = hsize / st, ow = wsize / st, gch = och / S, b;
	double delj;

	//誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し) ghost
	if (rel) BReLUBack(outdelta, out, gch, och, oh, ow, BS);
	if (CBatchNormal) for (j = gch; j < och; j++) //BNするならここ
		CBatchNormalB(outdelta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], lr, oh, ow);

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < gch; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;						//  0-padding
	for (i = 0; i < gch; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPdelta[i][b][h][w] = 0;						//  0-padding
	for (i = 0; i < gch; i++)for (b = 0; b < BS; b++)for (h = 0; h < oh; h++)for (w = 0; w < ow; w++) BPin[i][b][h + p][w + p] = out[i][b][h][w];	// データ入力

	//線形変換処理(DepthWise畳み込み)
	for (i = 0, j = gch; j < och; i++, j++) { // 各入力チャネルを処理
		if (i >= gch) i = 0;
		for (b = 0; b < BS; b++)
			for (h = 0; h < oh; h++) for (w = 0; w < ow; w++) {
				dbias[j] += outdelta[j][b][h][w]; // 畳み込みの合計値を初期化
				for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {
					dwoi[j][0][y][x] += BPin[i][b][h + y][w + x] * outdelta[j][b][h][w];
					BPdelta[i][b][h + y][w + x] += woi[j][0][y][x] * outdelta[j][b][h][w];// 勾配計算
				}
			}
	}
	// ghost deltaを上乗せ
	for (i = 0; i < gch; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++)
		outdelta[i][b][h][w] += BPdelta[i][b][h + p][w + p];	// デルタのパディング削除

	//作られた誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し) not ghost
	if (rel) BReLUBack(outdelta, out, 0, gch, oh, ow, BS);
	if (CBatchNormal) for (j = 0; j < gch; j++) //BNするならここ
		CBatchNormalB(outdelta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], lr, oh, ow);

	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPdelta[i][b][h][w] = 0;						// デルタ 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;							// 入力 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];	// データ入力

	// back convolution
	for (j = 0; j < gch; j++)for (b = 0; b < BS; b++) {//出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = outdelta[j][b][h / st][w / st];
			for (dbias[j] += delj, i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++) {//ここにｙとｘのループ
					//前向きで使ったところと同じ関係で誤差信号と重みの修正量を蓄積させる
					//out[j][h][w] += woi[j][i][y][x] * Pin[i][h * st + y][w * st + x];
					BPdelta[i][b][h + y][w + x] += woi[j][i][y][x] * delj;
					dwoi[j][i][y][x] += BPin[i][b][h + y][w + x] * delj;
				}
		}
	}

	// in デルタのパディング戻し
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		indelta[i][b][h][w] = BPdelta[i][b][h + p][w + p];
}

void CBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1;
	int y, x, n, h, w, my, mx;
	double max;
	//重み修正量を０初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dwoi[s][c][j][i][y][x] = 0; //0

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;
		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = out[s][c][j][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						delta[s][c][j][h + y][w + x] = 0;
						if (max < out[s][c][j][h + y][w + x]) {
							max = out[s][c][j][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					if (out[s][c][j][h + my][w + mx] > 0) delta[s][c][j][h + my][w + mx] = mdelta[n];
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			for (j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					max = out[s][c][j][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						delta[s][c][j][h + y][w + x] = 0;
						if (max < out[s][c][j][h + y][w + x]) {
							max = out[s][c][j][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					if (out[s][c][j][h + my][w + mx] > 0)
						delta[s][c][j][h + my][w + mx] = delta[s + 1][0][j][h / PS][w / PS];
				}
			}
		}
		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			BackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1],
				hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], 1);
		}
	}

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H, wsize = W, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS)
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / (double)w;
}
void BCBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr,
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1;
	int y, x, n, h, w, my, mx, b;
	double max;
	//重み修正量を０初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dwoi[s][c][j][i][y][x] = 0;

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;
		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (b = 0; b < BS; b++)
				for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
					for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
						max = out[s][c][j][b][h][w]; my = 0; mx = 0;
						for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
							delta[s][c][j][b][h + y][w + x] = 0;
							if (max < out[s][c][j][b][h + y][w + x]) {
								max = out[s][c][j][b][h + y][w + x];
								my = y;
								mx = x;
							}
						}
						delta[s][c][j][b][h + my][w + mx] = mdelta[n][b];
					}
				}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			for (j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS) {
					max = out[s][c][j][b][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						delta[s][c][j][b][h + y][w + x] = 0;
						if (max < out[s][c][j][b][h + y][w + x]) {
							max = out[s][c][j][b][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					delta[s][c][j][b][h + my][w + mx] = delta[s + 1][0][j][b][h / PS][w / PS];
				}
			}
		}
		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			BBackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1],
				hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], bnet[s][c],
				mean[s][c], var[s][c], gamma[s][c], beta[s][c], lr, 1);
		}
	}

	// 重み更新
	for (hsize = H, wsize = W, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS)
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / (double)w;
}

void SkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, 
	double in[ICH][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][H][W], double skipdelta[SEG][CHMAX][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS]) {
	int i, j, s, c, hsize, wsize, st = 1, y, x, n, h, w, my, mx, och;
	double max, sum, div;
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	//重み修正量を０初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dwoi[s][c][j][i][y][x] = 0; //0

	for (c = CONV, s = 0; s < SEG; s++)
		for (j = 0; j < CHMAX; j++) for (dskipbias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dskipwoi[s][j][i][y][x] = 0; //0

	for (j = 0; j < chN[0][0]; j++) for (dibias[j] = 0, i = 0; i < ICH; i++)
		for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++) diwoi[j][i][y][x] = 0; //0

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;
		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = skipout[s][j][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						skipdelta[s][j][h + y][w + x] = 0;
						if (max < skipout[s][j][h + y][w + x]) {
							max = skipout[s][j][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					//if (skipout[s][j][h + my][w + mx] > 0) 
					skipdelta[s][j][h + my][w + mx] = mdelta[n];
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			BackConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0], 1);
			for (j = 0; j < chN[s][c]; j++)for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS)
				for (y = 0; y < PS; y++)for (x = 0; x < PS; x++)
					skipdelta[s][j][h + y][w + x] += delta[s + 1][0][j][h / PS][w / PS] / (PS * PS);//skip-connection
		}
		for (j = 0; j < chN[s][c]; j++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][c][j][h][w] = skipdelta[s][j][h][w];

		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			BackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], 1);
		}
		for (j = 0; j < chN[s][0]; j++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][0][j][h][w] += skipdelta[s][j][h][w];//skip-connection
	}
	InBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dibias, diwoi, out[0][0]);

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (div = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / div, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / div;
		div /= PS * PS;
		if (s == SEG - 1) och = CHMAX; else och = chN[s + 1][0];
		for (j = 0; j < och; j++)
			for (skipbias[s][j] += -lr * dskipbias[s][j] / div, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipwoi[s][j][i][y][x] += -lr * dskipwoi[s][j][i][y][x] / div;

	}
	div = (H - IKS + 1) * (W - IKS + 1);
	for (j = 0; j < chN[0][0]; j++)
		for (inbias[j] += -lr * dibias[j] / div, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inwoi[j][i][y][x] += -lr * diwoi[j][i][y][x] / div; //0
}
void BSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[NMAX][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr,
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][BS][H][W], double skipdelta[SEG][CHMAX][BS][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX],
	double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS]) {
	int i, j, s, c, hsize, wsize, st = 1, b, y, x, n, h, w, my, mx, och;
	double max, div;
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	//重み修正量を０初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dwoi[s][c][j][i][y][x] = 0; //0

	for (c = CONV, s = 0; s < SEG; s++)
		for (j = 0; j < CHMAX; j++) for (dskipbias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dskipwoi[s][j][i][y][x] = 0; //0

	for (j = 0; j < chN[0][0]; j++) for (dibias[j] = 0, i = 0; i < ICH; i++)
		for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++) diwoi[j][i][y][x] = 0; //0

	//処理
	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;
		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (n = 0, j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++) {//出力側
				for (n = j * hsize * wsize / (PS * PS), h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = skipout[s][j][b][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						skipdelta[s][j][b][h + y][w + x] = 0;
						if (max < skipout[s][j][b][h + y][w + x]) {
							max = skipout[s][j][b][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					//if (skipout[s][j][h + my][w + mx] > 0) 
					skipdelta[s][j][b][h + my][w + mx] = mdelta[n][b];
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			//通常時
			BBackConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0], bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], lr, 1);
			for (j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS)
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++)
						skipdelta[s][j][b][h + y][w + x] += delta[s + 1][0][j][b][h / PS][w / PS] / (PS * PS);
		}
		for (j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][c][j][b][h][w] = skipdelta[s][j][b][h][w];

		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			BBackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], lr, 1);
		}
		for (j = 0; j < chN[s][0]; j++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][0][j][b][h][w] += skipdelta[s][j][b][h][w];//skip-connection
	}
	BInBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dibias, diwoi, out[0][0], bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], lr);

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (div = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / div, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / div;
		div /= PS * PS;
		if (s == SEG - 1) och = CHMAX; else och = chN[s + 1][0];
		for (j = 0; j < och; j++)
			for (skipbias[s][j] += -lr * dskipbias[s][j] / div, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipwoi[s][j][i][y][x] += -lr * dskipwoi[s][j][i][y][x] / div;
	}
	div = (H - IKS + 1) * (W - IKS + 1) * BS;
	for (j = 0; j < chN[0][0]; j++)
		for (inbias[j] += -lr * dibias[j] / div, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inwoi[j][i][y][x] += -lr * diwoi[j][i][y][x] / div; //0
}

void BackGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS]) {
	int i, h, w;

	//	//2回目GhostModule→出力をgoutへ格納
	//	GhostConv(och, och, gbias, gwoi, gin, gout, hsize/st, wsize/st, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);
	BackGhostConv(och, och, gbias, gwoi, Gapdelta, delta, hsize / st, wsize / st, 1, KS, gin, dgbias, dgwoi, gout, 0);

	//1回目
	//GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);
	BackGhostConv(ich, och, bias, woi, indelta, Gapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
		indelta[i][h][w] += delta[i][h / st][w / st];
}
void BBackGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][BS][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double lr) {
	int i, h, w, b;

	//	//2回目GhostModule→出力をgoutへ格納
	//BGhostConv(och, och, gbias, gwoi, gin, gout, oh, ow, 1, KS, gbnet, gmean, gvar, ggamma, gbeta, gmmean, gmvar, 0);
	BBackGhostConv(och, och, gbias, gwoi, BGapdelta, delta, hsize / st, wsize / st, 1, KS, gin, dgbias, dgwoi, gout, gbnet, gmean, gvar, ggamma, gbeta, lr, 0);

	//1回目
	//BGhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, bnet, mean, var, gamma, beta, mmean, mvar, 1);
	BBackGhostConv(ich, och, bias, woi, indelta, BGapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, bnet, mean, var, gamma, beta, lr, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			indelta[i][b][h][w] += delta[i][b][h / st][w / st];
}

void GBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, 
	double in[ICH][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][H][W], double skipdelta[SEG][CHMAX][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[][CONV][CHMAX][H][W], double gout[][CONV][CHMAX][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1;
	int y, x, n, h, w, my, mx, och;
	double max;
	//重み修正量を０初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dwoi[s][c][j][i][y][x] = 0; //0

	for (c = CONV, s = 0; s < SEG; s++)
		for (j = 0; j < CHMAX; j++)
			for (dskipbias[s][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dskipwoi[s][j][i][y][x] = 0; //0

	for (j = 0; j < chN[0][0]; j++)
		for (dibias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				diwoi[j][i][y][x] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dgbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dgwoi[s][c][j][i][y][x] = 0; //0

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;

		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = skipout[s][j][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						skipdelta[s][j][h + y][w + x] = 0;
						if (max < skipout[s][j][h + y][w + x]) {
							max = skipout[s][j][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					//if (skipout[s][j][h + my][w + mx] > 0) 
					skipdelta[s][j][h + my][w + mx] = mdelta[n];
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			//通常時
			BackConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0], 1);
			//Ghost時
			//BackGhostConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0]);
			for (j = 0; j < chN[s][c]; j++)for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS)
				for (y = 0; y < PS; y++)for (x = 0; x < PS; x++)
					skipdelta[s][j][h + y][w + x] += delta[s + 1][0][j][h / PS][w / PS] / (PS * PS);
		}
		for (j = 0; j < chN[s][c]; j++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][c][j][h][w] = skipdelta[s][j][h][w];

		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			if (s == 0) BackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], 1);
			//else BackGhostConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], 1);
			else
				BackGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1],
					gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c]);
		}
		if (s == 0)
			for (j = 0; j < chN[s][0]; j++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
				delta[s][0][j][h][w] += skipdelta[s][j][h][w];//skip-connection
	}
	InBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dibias, diwoi, out[0][0]);

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / (double)w;
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (gbias[s][c][j] += -lr * dgbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					gwoi[s][c][j][i][y][x] += -lr * dgwoi[s][c][j][i][y][x] / (double)w;
		w /= PS * PS;
		if (s == SEG - 1) och = CHMAX; else och = chN[s + 1][0];
		for (j = 0; j < och; j++)
			for (skipbias[s][j] += -lr * dskipbias[s][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipwoi[s][j][i][y][x] += -lr * dskipwoi[s][j][i][y][x] / (double)w;

	}
	w = (H - IKS + 1) * (W - IKS + 1);
	for (j = 0; j < chN[0][0]; j++)
		for (inbias[j] += -lr * dibias[j] / (double)w, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inwoi[j][i][y][x] += -lr * diwoi[j][i][y][x] / (double)w; //0
}
void BGBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[NMAX][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, 
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][BS][H][W], double skipdelta[SEG][CHMAX][BS][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS],
	double gbnet[][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1, b;
	int y, x, n, h, w, my, mx, och;
	double max;
	//重み修正量を０初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dwoi[s][c][j][i][y][x] = 0; //0

	for (c = CONV, s = 0; s < SEG; s++)
		for (j = 0; j < CHMAX; j++)
			for (dskipbias[s][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dskipwoi[s][j][i][y][x] = 0; //0

	for (j = 0; j < chN[0][0]; j++)
		for (dibias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				diwoi[j][i][y][x] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dgbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dgwoi[s][c][j][i][y][x] = 0; //0

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;
		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (b = 0; b < BS; b++) for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = skipout[s][j][b][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						skipdelta[s][j][b][h + y][w + x] = 0;
						if (max < skipout[s][j][b][h + y][w + x]) {
							max = skipout[s][j][b][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					//if (skipout[s][j][h + my][w + mx] > 0) 
					skipdelta[s][j][b][h + my][w + mx] = mdelta[n][b];
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			//通常時
			BBackConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0], bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], lr, 1);
			for (j = 0; j < chN[s][c]; j++) for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS)
					for (y = 0; y < PS; y++) for (x = 0; x < PS; x++)
						skipdelta[s][j][b][h + y][w + x] += delta[s + 1][0][j][b][h / PS][w / PS] / (PS * PS);
		}
		for (j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][c][j][b][h][w] = skipdelta[s][j][b][h][w];//skip-connection

		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			if (s == 0) BBackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], lr, 1);
			//else BBackGhostConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], lr,1);
			else
				BBackGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1],
					gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
					bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], lr);
		}
		if (s == 0)
			for (j = 0; j < chN[s][0]; j++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
				delta[s][0][j][b][h][w] += skipdelta[s][j][b][h][w];//skip-connection
	}
	BInBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dibias, diwoi, out[0][0], bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], lr);

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / (double)w;
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (gbias[s][c][j] += -lr * dgbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					gwoi[s][c][j][i][y][x] += -lr * dgwoi[s][c][j][i][y][x] / (double)w;
		w /= PS * PS;
		if (s == SEG - 1) och = CHMAX; else och = chN[s + 1][0];
		for (j = 0; j < och; j++)
			for (skipbias[s][j] += -lr * dskipbias[s][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipwoi[s][j][i][y][x] += -lr * dskipwoi[s][j][i][y][x] / (double)w;
	}
	w = (H - IKS + 1) * (W - IKS + 1) * BS;
	for (j = 0; j < chN[0][0]; j++)
		for (inbias[j] += -lr * dibias[j] / (double)w, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inwoi[j][i][y][x] += -lr * diwoi[j][i][y][x] / (double)w; //0
}

void BackPConv(int ich, int och, double bias[], double woi[][CHMAX], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, double dbias[], double dwoi[][CHMAX], double in[][H][W], double out[][H][W], int rel) {
	int i, j, h, w, oh = hsize / st, ow = wsize / st;
	double delj;
	// 初期化
	for (i = 0; i < ich; i++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) indelta[i][h][w] = 0.0;

	// 出力側誤差信号にReLUの微分を適用
	if (rel)  ReLUBack(delta, out, 0, och, oh, ow);

	// back 1x1 convolution
	for (j = 0; j < och; j++) { // 出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = delta[j][h / st][w / st];
			for (dbias[j] += delj, i = 0; i < ich; i++) {// バイアスの更新 // 入力側				
				indelta[i][h][w] += woi[j][i] * delj;// 1x1 の畳み込みに基づく誤差逆伝播				
				dwoi[j][i] += in[i][h][w] * delj;// 重みの更新量を計算
			}
		}
	}
}
void BBackPConv(int ich, int och, double bias[], double woi[][CHMAX], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize,
	int st, double dbias[], double dwoi[][CHMAX], double in[][BS][H][W], double out[][BS][H][W], int rel) {
	int i, j, h, w, b, oh = hsize / st, ow = wsize / st;
	double delj;
	// 初期化
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) indelta[i][b][h][w] = 0.0;

	// 出力側誤差信号にReLUの微分を適用
	if (rel) BReLUBack(delta, out, 0, och, oh, ow, BS);
	//if (CBatchNormal) for (j = 0; j < och; j++) //BNするならここ
	//	CBatchNormalB(delta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], lr, oh, ow);

	// back 1x1 convolution
	for (j = 0; j < och; j++)for (b = 0; b < BS; b++) { // 出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = delta[j][b][h / st][w / st];
			for (dbias[j] += delj, i = 0; i < ich; i++) {// バイアスの更新 // 入力側					
				indelta[i][b][h][w] += woi[j][i] * delj;// 1x1 の畳み込みに基づく誤差逆伝播					
				dwoi[j][i] += in[i][b][h][w] * delj;// 重みの更新量を計算
			}
		}
	}
}
void BackCatPConv(int ich, int och, double bias[], double woi[][CHMAX * 2], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, double dbias[], double dwoi[][CHMAX * 2], double in[][H][W], double out[][H][W], int rel) {
	int i, j, h, w, oh = hsize / st, ow = wsize / st;
	double delj;
	// 初期化
	for (i = 0; i < ich; i++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) indelta[i][h][w] = 0.0;

	// 出力側誤差信号にReLUの微分を適用
	if (rel)  ReLUBack(delta, out, 0, och, oh, ow);

	// back 1x1 convolution
	for (j = 0; j < och; j++) { // 出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = delta[j][h / st][w / st];
			for (dbias[j] += delj, i = 0; i < ich; i++) {// バイアスの更新 // 入力側				
				indelta[i][h][w] += woi[j][i] * delj;// 1x1 の畳み込みに基づく誤差逆伝播				
				dwoi[j][i] += in[i][h][w] * delj;// 重みの更新量を計算
			}
		}
	}
}
void BBackCatPConv(int ich, int och, double bias[], double woi[][CHMAX * 2], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize,
	int st, double dbias[], double dwoi[][CHMAX * 2], double in[][BS][H][W], double out[][BS][H][W], int rel) {
	int i, j, h, w, b, oh = hsize / st, ow = wsize / st;
	double delj;
	// 初期化
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) indelta[i][b][h][w] = 0.0;

	// 出力側誤差信号にReLUの微分を適用
	if (rel) BReLUBack(delta, out, 0, och, oh, ow, BS);
	//if (CBatchNormal) for (j = 0; j < och; j++) //BNするならここ
	//	CBatchNormalB(delta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], lr, oh, ow);

	// back 1x1 convolution
	for (j = 0; j < och; j++)for (b = 0; b < BS; b++) { // 出力側
		for (h = 0; h < hsize; h += st) for (w = 0; w < wsize; w += st) {
			delj = delta[j][b][h / st][w / st];
			for (dbias[j] += delj, i = 0; i < ich; i++) {// バイアスの更新 // 入力側					
				indelta[i][b][h][w] += woi[j][i] * delj;// 1x1 の畳み込みに基づく誤差逆伝播					
				dwoi[j][i] += in[i][b][h][w] * delj;// 重みの更新量を計算
			}
		}
	}
}

void BackCAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double indelta[][H][W], double delta[][H][W], int hsize, int wsize, double dbias[][CHMAX], double dwoi[][CHMAX][CHMAX], double in[][H][W], double out[][H][W],
	double cain[][H][W], double caout[][H][W], double gapin[][H][W]) {

	int i, h, w;
	double gap_sum, div=(double)(hsize*wsize);

	// 1. 出力誤差を処理前と処理後の積から逆伝搬 (ReLU戻しと誤差計算)
	//for (i = 0; i < ich; i++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
	//	out[i][h][w] = in[i][h][w] * caout[i][0][0];// 処理前と処理後の積計算
	for (i = 0; i < ich; i++) {
		for (CS1delta[i][0][0] = 0, h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			CS1delta[i][0][0] += in[i][h][w] * delta[i][h][w];
		//CS1delta[i][0][0] /= div;
	}

	// 2. シグモイドの微分を適用してcaoutの誤差を逆伝搬
	//for (i = 0; i < ich; i++) caout[i][0][0] = 1.0 / (1.0 + exp(-caout[i][0][0])); //sigmoid
	for (i = 0; i < ich; i++) CS1delta[i][0][0] *= caout[i][0][0] * (1.0 - caout[i][0][0]); // シグモイドの微分

	// 3. 2回目の逆伝搬
	//PConv(ich, ich, bias[1], woi[1], cain, caout, 1, 1, gamma, beta, mmean, mvar, 0);
	BackPConv(ich, ich, bias[1], woi[1], CS2delta, CS1delta, 1, 1, 1, dbias[1], dwoi[1], cain, caout, 0);

	// 4. 1回目の逆伝搬
	//PConv(ich, ich, bias[0], woi[0], gapin, cain, 1, 1, gamma, beta, mmean, mvar, 1);
	BackPConv(ich, ich, bias[0], woi[0], Gapdelta, CS2delta, 1, 1, 1, dbias[0], dwoi[0], gapin, cain, 1);

	// 5. Global Average Poolingの逆伝搬
	for (i = 0; i < ich; i++) {
		gap_sum = Gapdelta[i][0][0] / div; // GAPで計算された誤差
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			indelta[i][h][w] = gap_sum; // GAPの逆伝搬
	}
}
void BBackCAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize, double dbias[][CHMAX], double dwoi[][CHMAX][CHMAX],
	double mean[][CHMAX], double var[][CHMAX], double gamma[][CHMAX], double beta[][CHMAX], double in[][BS][H][W], double out[][BS][H][W],
	double cain[][BS][H][W], double caout[][BS][H][W], double gapin[][BS][H][W]) {

	int i, h, w, b;
	double gap_sum, div = (double)hsize * wsize;

	// 1. 出力誤差を処理前と処理後の積から逆伝搬 (ReLU戻しと誤差計算)
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++){
			for (BCS1delta[i][b][0][0] = 0, h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
				BCS1delta[i][b][0][0] += in[i][b][h][w] * delta[i][b][h][w];
			}
			//BCS1delta[i][b][0][0] /= div;
	}

	// 2. シグモイドの微分を適用してcaoutの誤差を逆伝搬
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++)
		BCS1delta[i][b][0][0] *= caout[i][b][0][0] * (1.0 - caout[i][b][0][0]); // シグモイドの微分

	// 3. 2回目の逆伝搬
	BBackPConv(ich, ich, bias[1], woi[1], BCS2delta, BCS1delta, 1, 1, 1, dbias[1], dwoi[1], cain, caout, 0);

	// 4. 1回目の逆伝搬
	BBackPConv(ich, ich, bias[0], woi[0], BGapdelta, BCS2delta, 1, 1, 1, dbias[0], dwoi[0], gapin, cain, 1);

	// 5. Global Average Poolingの逆伝搬
	for (i = 0; i < ich; i++) for (b = 0; b < BS; b++) {
		gap_sum = BGapdelta[i][b][0][0] / div; // GAPで計算された誤差→(gapsumの配列要確認)
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			indelta[i][b][h][w] = gap_sum; // GAPの逆伝搬
	}
}

void BackSAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double indelta[][H][W], double delta[][H][W], int hsize, int wsize, double dbias[][CHMAX], double dwoi[][CHMAX][CHMAX], double in[][H][W], double out[][H][W],
	double sain[][H][W], double saout[][H][W]) {

	int j, h, w;

	//初期化
	for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++) CS1delta[0][h][w] = 0;

	// 5. 入力の誤差を処理
	for (j = 0; j < ich; j++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		CS1delta[0][h][w] += in[j][h][w] * delta[j][h][w];

	// 2. シグモイドの微分を適用してsaoutの誤差を逆伝搬
	for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		CS1delta[0][h][w] *= saout[0][h][w] * (1.0 - saout[0][h][w]); // シグモイドの微分
		//CS1delta[0][h][w] *= saout[0][h][w] * (1.0 - saout[0][h][w]) / ich; // シグモイドの微分

	// 3. 2回目のCONV層の逆伝搬
	BackPConv(ich, 1, bias[1], woi[1], CS2delta, CS1delta, hsize, wsize, 1, dbias[1], dwoi[1], sain, saout, 0);

	// 4. 1回目のCONV層の逆伝搬
	BackPConv(ich, ich, bias[0], woi[0], indelta, CS2delta, hsize, wsize, 1, dbias[0], dwoi[0], in, sain, 1);
}
void BBackSAUnit(int ich, double bias[][CHMAX], double woi[][CHMAX][CHMAX], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize, double dbias[][CHMAX], double dwoi[][CHMAX][CHMAX],
	double mean[][CHMAX], double var[][CHMAX], double gamma[][CHMAX], double beta[][CHMAX], double in[][BS][H][W], double out[][BS][H][W], double sain[][BS][H][W], double saout[][BS][H][W]) {

	int j, h, w, b;

	//初期化
	for (b = 0; b < BS; b++) for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		BCS1delta[0][b][h][w] = 0;

	// 5. 入力の誤差を処理
	for (j = 0; j < ich; j++) for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
			BCS1delta[0][b][h][w] += in[j][b][h][w] * delta[j][b][h][w];

	// 2. シグモイドの微分を適用してsaoutの誤差を逆伝搬
	for (b = 0; b < BS; b++) for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
		BCS1delta[0][b][h][w] *= saout[0][b][h][w] * (1.0 - saout[0][b][h][w]); // シグモイドの微分
		//BCS1delta[0][b][h][w] *= saout[0][b][h][w] * (1.0 - saout[0][b][h][w]) / ich; // シグモイドの微分

	// 3. 2回目のCONV層の逆伝搬
	BBackPConv(ich, 1, bias[1], woi[1], BCS2delta, BCS1delta, hsize, wsize, 1, dbias[1], dwoi[1], sain, saout, 0);

	// 4. 1回目のCONV層の逆伝搬
	BBackPConv(ich, ich, bias[0], woi[0], indelta, BCS2delta, hsize, wsize, 1, dbias[0], dwoi[0], in, sain, 1);
}

void BackCSGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize, int st, int ksize, double in[][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS],
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][H][W], double cunitout[][H][W], double gapin[][H][W],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][H][W], double sunitout[][H][W], double resbias[], double reswoi[][CHMAX * 2], double resout[][H][W],
	double dcabias[][CHMAX], double dcawoi[][CHMAX][CHMAX], double dsabias[][CHMAX], double dsawoi[][CHMAX][CHMAX], double dresbias[], double dreswoi[][CHMAX * 2]) {

	int i, j, h, w, oh = hsize / st, ow = wsize / st;

	//PCONV
	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			ConcatOut[j][h][w] = cunitout[j][h][w];
	}
	for (; j < och + och / S; j++) {
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			ConcatOut[j][h][w] = sunitout[j - och][h][w];
	}
	BackCatPConv(och + och / S, och, resbias, reswoi, ConcatDelta, delta, oh, ow, 1, dresbias, dreswoi, ConcatOut, resout, 0);

	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			CunitDelta[j][h][w] = ConcatDelta[j][h][w];
	}
	for (; j < och + och / S; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			SunitDelta[j - och][h][w] = ConcatDelta[j][h][w];
	}
	//CAUnit(och, cabias, cawoi, gout, cunitout, oh, ow, cagamma, cabeta, cammean, camvar, cain, caout, gapin);
	BackCAUnit(och, cabias, cawoi, CaGDelta, CunitDelta, oh, ow, dcabias, dcawoi, gout, cunitout, caout[0], caout[1], gapin);
	BackSAUnit(och / S, sabias, sawoi, SaGDelta, SunitDelta, oh, ow, dsabias, dsawoi, gout, sunitout, saout[0], saout[1]);

	//for (j = 0; j < och; j++) {
	//	for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
	//		gdelta[j][h][w]=CaGDelta[j][h][w] + SaGDelta[j][h][w];
	//}
	for (j = 0; j < och / S; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			CaGDelta[j][h][w] += SaGDelta[j][h][w];
	}

	//2回目GhostModule→出力をgoutへ格納
	BackGhostConv(och, och, gbias, gwoi, Gapdelta, CaGDelta, oh, ow, 1, KS, gin, dgbias, dgwoi, gout, 0);

	//1回目
	BackGhostConv(ich, och, bias, woi, indelta, Gapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		indelta[i][h][w] += delta[i][h / st][w / st];
}
void BBackCSGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize, int st, int ksize, double in[][BS][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][BS][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double lr,
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][BS][H][W], double cunitout[][BS][H][W], double gapin[][BS][H][W], double camean[][CHMAX], double cavar[][CHMAX], double cagamma[][CHMAX], double cabeta[][CHMAX],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][BS][H][W], double sunitout[][BS][H][W], double samean[][CHMAX], double savar[][CHMAX], double sagamma[][CHMAX], double sabeta[][CHMAX],
	double resbias[], double reswoi[][CHMAX * 2], double resout[][BS][H][W],
	double dcabias[][CHMAX], double dcawoi[][CHMAX][CHMAX], double dsabias[][CHMAX], double dsawoi[][CHMAX][CHMAX], double dresbias[], double dreswoi[][CHMAX * 2]) {

	int i, j, h, w, oh = hsize / st, ow = wsize / st, b;

	//PCONV
	for (j = 0; j < och; j++) for (b = 0; b < BS; b++)//(要確認)
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			BConcatOut[j][b][h][w] = cunitout[j][b][h][w];

	for (; j < och + och / S; j++) for (b = 0; b < BS; b++)//(要確認)
		for (h = 0; h < oh; h++) for (w = 0; w < ow; w++)
			BConcatOut[j][b][h][w] = sunitout[j - och][b][h][w];
	BBackCatPConv(och + och / S, och, resbias, reswoi, BConcatDelta, delta, oh, ow, 1, dresbias, dreswoi, BConcatOut, resout, 0);

	for (j = 0; j < och; j++) for (b = 0; b < BS; b++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			BCunitDelta[j][b][h][w] = BConcatDelta[j][b][h][w];
	}
	for (; j < och + och / S; j++) for (b = 0; b < BS; b++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			BSunitDelta[j - och][b][h][w] = BConcatDelta[j][b][h][w];
	}
	BBackCAUnit(och, cabias, cawoi, BCaGDelta, BCunitDelta, oh, ow, dcabias, dcawoi, camean, cavar, cagamma, cabeta, gout, cunitout, caout[0], caout[1], gapin);
	BBackSAUnit(och / S, sabias, sawoi, BSaGDelta, BSunitDelta, oh, ow, dsabias, dsawoi, samean, savar, sagamma, sabeta, gout, sunitout, saout[0], saout[1]);

	//for (j = 0; j < och; j++) for (b = 0; b < BS; b++)
	//	for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
	//		gdelta[j][b][h][w] = BCaGDelta[j][b][h][w] + BSaGDelta[j][b][h][w];
	for (j = 0; j < och / S; j++) for (b = 0; b < BS; b++)
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			BCaGDelta[j][b][h][w] += BSaGDelta[j][b][h][w];

	//	//2回目GhostModule→出力をgoutへ格納
	//	GhostConv(och, och, gbias, gwoi, gin, gout, hsize/st, wsize/st, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);
	BBackGhostConv(och, och, gbias, gwoi, BGapdelta, BCaGDelta, oh, ow, 1, KS, gin, dgbias, dgwoi, gout, gbnet, gmean, gvar, ggamma, gbeta, lr, 0);
	//1回目
	//GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);
	BBackGhostConv(ich, och, bias, woi, indelta, BGapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, bnet, mean, var, gamma, beta, lr, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
		indelta[i][b][h][w] += delta[i][b][h / st][w / st];
}

void BackCGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize, int st, int ksize, double in[][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS],
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][H][W], double cunitout[][H][W], double gapin[][H][W], double dcabias[][CHMAX], double dcawoi[][CHMAX][CHMAX]) {

	int i, j, h, w, oh = hsize / st, ow = wsize / st;

	//CAUnit(och, cabias, cawoi, gout, cunitout, oh, ow, cagamma, cabeta, cammean, camvar, cain, caout, gapin);
	BackCAUnit(och, cabias, cawoi, CaGDelta, delta, oh, ow, dcabias, dcawoi, gout, cunitout, caout[0], caout[1], gapin);
	//BackSAUnit(och / S, sabias, sawoi, SaGDelta, sunitdelta, oh, ow, dsabias, dsawoi, gout, sunitout, saout[0], saout[1]);


	//	//2回目GhostModule→出力をgoutへ格納
	BackGhostConv(och, och, gbias, gwoi, Gapdelta, CaGDelta, oh, ow, 1, KS, gin, dgbias, dgwoi, gout, 0);

	//1回目
	BackGhostConv(ich, och, bias, woi, indelta, Gapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		indelta[i][h][w] += delta[i][h / st][w / st];
}
void BBackCGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize, int st, int ksize, double in[][BS][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][BS][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double lr, 
	double cabias[][CHMAX], double cawoi[][CHMAX][CHMAX], double caout[][CHMAX][BS][H][W], double cunitout[][BS][H][W], double gapin[][BS][H][W], double camean[][CHMAX], double cavar[][CHMAX], double cagamma[][CHMAX], double cabeta[][CHMAX],
	double dcabias[][CHMAX], double dcawoi[][CHMAX][CHMAX]) {

	int i, j, h, w, oh = hsize / st, ow = wsize / st, b;

	BBackCAUnit(och, cabias, cawoi, BCaGDelta, delta, oh, ow, dcabias, dcawoi, camean, cavar, cagamma, cabeta, gout, cunitout, caout[0], caout[1], gapin);

	//	//2回目GhostModule→出力をgoutへ格納
	//	GhostConv(och, och, gbias, gwoi, gin, gout, hsize/st, wsize/st, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);
	BBackGhostConv(och, och, gbias, gwoi, BGapdelta, BCaGDelta, oh, ow, 1, KS, gin, dgbias, dgwoi, gout, gbnet, gmean, gvar, ggamma, gbeta, lr, 0);
	//1回目
	//GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);
	BBackGhostConv(ich, och, bias, woi, indelta, BGapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, bnet, mean, var, gamma, beta, lr, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
		indelta[i][b][h][w] += delta[i][b][h / st][w / st];
}

void BackSGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize, int st, int ksize, double in[][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][H][W], double gout[][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS],
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][H][W], double sunitout[][H][W], double dsabias[][CHMAX], double dsawoi[][CHMAX][CHMAX]) {

	int i, j, h, w, oh = hsize / st, ow = wsize / st;

	BackSAUnit(och, sabias, sawoi, SaGDelta, delta, oh, ow, dsabias, dsawoi, gout, sunitout, saout[0], saout[1]);

	//	//2回目GhostModule→出力をgoutへ格納
	BackGhostConv(och, och, gbias, gwoi, Gapdelta, SaGDelta, oh, ow, 1, KS, gin, dgbias, dgwoi, gout, 0);

	//1回目
	BackGhostConv(ich, och, bias, woi, indelta, Gapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		indelta[i][h][w] += delta[i][h / st][w / st];
}
void BBackSGhostbottleneck(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize, int st, int ksize, double in[][BS][H][W], double dbias[], double dwoi[][CHMAX][KS][KS], double out[][BS][H][W],
	double gbias[], double gwoi[][CHMAX][KS][KS], double gin[][BS][H][W], double gout[][BS][H][W], double dgbias[], double dgwoi[][CHMAX][KS][KS], double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double gbnet[][BS][H][W], double gmean[], double gvar[], double ggamma[], double gbeta[], double lr,
	double sabias[][CHMAX], double sawoi[][CHMAX][CHMAX], double saout[][CHMAX][BS][H][W], double sunitout[][BS][H][W], double samean[][CHMAX], double savar[][CHMAX], double sagamma[][CHMAX], double sabeta[][CHMAX], double dsabias[][CHMAX], double dsawoi[][CHMAX][CHMAX]) {

	int i, j, h, w, oh = hsize / st, ow = wsize / st, b;

	BBackSAUnit(och, sabias, sawoi, BSaGDelta, delta, oh, ow, dsabias, dsawoi, samean, savar, sagamma, sabeta, gout, sunitout, saout[0], saout[1]);

	//	//2回目GhostModule→出力をgoutへ格納
	//	GhostConv(och, och, gbias, gwoi, gin, gout, hsize/st, wsize/st, 1, KS, ggamma, gbeta, gmmean, gmvar, 0);
	BBackGhostConv(och, och, gbias, gwoi, BGapdelta, BSaGDelta, oh, ow, 1, KS, gin, dgbias, dgwoi, gout, gbnet, gmean, gvar, ggamma, gbeta, lr, 0);
	//1回目
	//GhostConv(ich, och, bias, woi, in, gin, hsize, wsize, st, KS, gamma, beta, mmean, mvar, 1);
	BBackGhostConv(ich, och, bias, woi, indelta, BGapdelta, hsize, wsize, st, KS, in, dbias, dwoi, gin, bnet, mean, var, gamma, beta, lr, 1);

	//skip-connection
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
		indelta[i][b][h][w] += delta[i][b][h / st][w / st];
}

void CSGBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, 
	double in[ICH][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][H][W], double skipdelta[SEG][CHMAX][H][W], double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[][CONV][CHMAX][H][W], double gout[][CONV][CHMAX][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][H][W], double cunitout[SEG][CONV][CHMAX][H][W], double gapin[SEG][CONV][CHMAX][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][H][W], double sunitout[SEG][CONV][CHMAX][H][W], double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][H][W],
	double dcabias[][CONV][2][CHMAX], double dcawoi[][CONV][2][CHMAX][CHMAX], double dsabias[][CONV][2][CHMAX], double dsawoi[][CONV][2][CHMAX][CHMAX], double dresbias[][CONV][CHMAX], double dreswoi[][CONV][CHMAX][CHMAX * 2]) {

	int i, j, s, c, p, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1;
	int y, x, n, h, w, my, mx, och;
	double max, gsum, gave;
	//重み修正量を０初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dwoi[s][c][j][i][y][x] = 0; //0
	for (c = CONV, s = 0; s < SEG; s++)
		for (j = 0; j < CHMAX; j++) for (dskipbias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dskipwoi[s][j][i][y][x] = 0; //0
	for (j = 0; j < chN[0][0]; j++) for (dibias[j] = 0, i = 0; i < ICH; i++)
		for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++) diwoi[j][i][y][x] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dgbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dgwoi[s][c][j][i][y][x] = 0; //0

//提案重み修正量を０初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (p = 0; p < 2; p++) for (j = 0; j < chN[s][c + 1]; j++)
			for (dcabias[s][c][p][j] = 0, i = 0; i < chN[s][c + 1]; i++) dcawoi[s][c][p][j][i] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (p = 0; p < 2; p++) for (j = 0; j < chN[s][c + 1]; j++)
			for (dsabias[s][c][p][j] = 0, i = 0; i < chN[s][c + 1]; i++) dsawoi[s][c][p][j][i] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dresbias[s][c][j] = 0, i = 0; i < chN[s][c + 1] * 2; i++)
			dreswoi[s][c][j][i] = 0; //0

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;

		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = skipout[s][j][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						skipdelta[s][j][h + y][w + x] = 0;
						if (max < skipout[s][j][h + y][w + x]) {
							max = skipout[s][j][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					//if (skipout[s][j][h + my][w + mx] > 0) 
					skipdelta[s][j][h + my][w + mx] = mdelta[n];
				}
			}
		}

		else {
			//通常時
			BackConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0], 1);
			//Ghost時
			//BackGhostConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0]);
			for (j = 0; j < chN[s][c]; j++)for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS)
				for (y = 0; y < PS; y++)for (x = 0; x < PS; x++)
					skipdelta[s][j][h + y][w + x] += delta[s + 1][0][j][h / PS][w / PS] / (PS * PS);
		}
		for (j = 0; j < chN[s][c]; j++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][c][j][h][w] = skipdelta[s][j][h][w];

		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			if (s == 0) BackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], 1);
			//else if (s == SEG - 1)BackGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1],
			//	gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gdelta[s][c], dgbias[s][c], dgwoi[s][c]);
			/*else BackCSGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c],
				dbias[s][c], dwoi[s][c], out[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
				cabias[s][c], cawoi[s][c], caout[s][c], cunitout[s][c], gapin[s][c], sabias[s][c], sawoi[s][c], saout[s][c], sunitout[s][c], resbias[s][c], reswoi[s][c], resout[s][c],
				dcabias[s][c], dcawoi[s][c], dsabias[s][c], dsawoi[s][c], dresbias[s][c], dreswoi[s][c]);*/
			/*else BackCGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c],
				dbias[s][c], dwoi[s][c], out[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
				cabias[s][c], cawoi[s][c], caout[s][c], cunitout[s][c], gapin[s][c],dcabias[s][c], dcawoi[s][c]);*/
			else BackSGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c],
				dbias[s][c], dwoi[s][c], out[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
				sabias[s][c], sawoi[s][c], saout[s][c], sunitout[s][c], dsabias[s][c], dsawoi[s][c]);
		}
		if (s == 0) for (j = 0; j < chN[s][0]; j++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][0][j][h][w] += skipdelta[s][j][h][w];//skip-connection
	}
	InBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dibias, diwoi, out[0][0]);

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / (double)w;
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (gbias[s][c][j] += -lr * dgbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					gwoi[s][c][j][i][y][x] += -lr * dgwoi[s][c][j][i][y][x] / (double)w;
		//以下3つ提案
		for (w = 1, c = 0; c < CONV; c++) for (p = 0; p < 2; p++) for (j = 0; j < chN[s][c + 1]; j++)
			for (cabias[s][c][p][j] += -lr * dcabias[s][c][p][j] / (double)w, i = 0; i < chN[s][c + 1]; i++)
				cawoi[s][c][p][j][i] += -lr * dcawoi[s][c][p][j][i] / (double)w;
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (p = 0; p < 2; p++) for (j = 0; j < chN[s][c + 1]; j++)
			for (sabias[s][c][p][j] += -lr * dsabias[s][c][p][j] / (double)w, i = 0; i < chN[s][c + 1]; i++)
				sawoi[s][c][p][j][i] += -lr * dsawoi[s][c][p][j][i] / (double)w;
		for (w = hsize * wsize, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (resbias[s][c][j] += -lr * dresbias[s][c][j] / (double)w, i = 0; i < chN[s][c + 1] * 2; i++)
				reswoi[s][c][j][i] += -lr * dreswoi[s][c][j][i] / (double)w;

		w /= PS * PS;
		if (s == SEG - 1) och = CHMAX; else och = chN[s + 1][0];
		for (j = 0; j < och; j++)
			for (skipbias[s][j] += -lr * dskipbias[s][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipwoi[s][j][i][y][x] += -lr * dskipwoi[s][j][i][y][x] / (double)w;

	}
	w = (H - IKS + 1) * (W - IKS + 1);
	for (j = 0; j < chN[0][0]; j++)
		for (inbias[j] += -lr * dibias[j] / (double)w, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inwoi[j][i][y][x] += -lr * diwoi[j][i][y][x] / (double)w; //0
}
void BCSGBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[NMAX][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, 
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][BS][H][W], double skipdelta[SEG][CHMAX][BS][H][W], double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS],
	double gbnet[][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], 
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][BS][H][W], double cunitout[SEG][CONV][CHMAX][BS][H][W], double gapin[SEG][CONV][CHMAX][BS][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][BS][H][W], double sunitout[SEG][CONV][CHMAX][BS][H][W], double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][BS][H][W],
	double camean[][CONV][2][CHMAX], double cavar[][CONV][2][CHMAX], double cagamma[][CONV][2][CHMAX], double cabeta[][CONV][2][CHMAX], double samean[][CONV][2][CHMAX], double savar[][CONV][2][CHMAX], double sagamma[][CONV][2][CHMAX], double sabeta[][CONV][2][CHMAX],
	double dcabias[][CONV][2][CHMAX], double dcawoi[][CONV][2][CHMAX][CHMAX], double dsabias[][CONV][2][CHMAX], double dsawoi[][CONV][2][CHMAX][CHMAX], double dresbias[][CONV][CHMAX], double dreswoi[][CONV][CHMAX][CHMAX * 2]) {

	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1, b, p;
	int y, x, n, h, w, my, mx, och;
	double max;
	//重み修正量を０初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++) for (dbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dwoi[s][c][j][i][y][x] = 0; //0
	for (c = CONV, s = 0; s < SEG; s++)
		for (j = 0; j < CHMAX; j++) for (dskipbias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++) dskipwoi[s][j][i][y][x] = 0; //0
	for (j = 0; j < chN[0][0]; j++)for (dibias[j] = 0, i = 0; i < ICH; i++)
		for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)diwoi[j][i][y][x] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)for (dgbias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)dgwoi[s][c][j][i][y][x] = 0; //0
//提案
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (p = 0; p < 2; p++)
		for (j = 0; j < chN[s][c + 1]; j++)for (dcabias[s][c][p][j] = 0, i = 0; i < chN[s][c + 1]; i++)
			dcawoi[s][c][p][j][i] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)for (p = 0; p < 2; p++)
		for (j = 0; j < chN[s][c + 1]; j++)for (dsabias[s][c][p][j] = 0, i = 0; i < chN[s][c + 1]; i++)
			dsawoi[s][c][p][j][i] = 0; //0
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
		for (j = 0; j < chN[s][c + 1]; j++)for (dresbias[s][c][j] = 0, i = 0; i < chN[s][c + 1] * 2; i++)
			dreswoi[s][c][j][i] = 0; //0

	for (s = SEG - 1; s >= 0; s--) {
		hsize *= PS;
		wsize *= PS;
		c = CONV;
		//pooling
		if (s == SEG - 1) { //out[s][c] -> mout 全結合に入れる
			for (b = 0; b < BS; b++) for (n = 0, j = 0; j < chN[s][c]; j++) {//出力側
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS, n++) {
					max = skipout[s][j][b][h][w]; my = 0; mx = 0;
					for (y = 0; y < PS; y++)for (x = 0; x < PS; x++) {//ここにｙとｘのループ
						skipdelta[s][j][b][h + y][w + x] = 0;
						if (max < skipout[s][j][b][h + y][w + x]) {
							max = skipout[s][j][b][h + y][w + x];
							my = y;
							mx = x;
						}
					}
					//if (skipout[s][j][h + my][w + mx] > 0) 
					skipdelta[s][j][b][h + my][w + mx] = mdelta[n][b];
				}
			}
		}
		else { //out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			//通常時
			BBackConv(chN[s][c], chN[s + 1][0], skipbias[s], skipwoi[s], skipdelta[s], delta[s + 1][0], hsize, wsize, PS, KS, skipout[s], dskipbias[s], dskipwoi[s], out[s + 1][0], bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], lr, 1);
			for (j = 0; j < chN[s][c]; j++) for (b = 0; b < BS; b++)
				for (h = 0; h < hsize; h += PS) for (w = 0; w < wsize; w += PS)
					for (y = 0; y < PS; y++) for (x = 0; x < PS; x++)
						skipdelta[s][j][b][h + y][w + x] += delta[s + 1][0][j][b][h / PS][w / PS] / (PS * PS);
		}
		for (j = 0; j < chN[s][c]; j++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][c][j][b][h][w] = skipdelta[s][j][b][h][w];//skip-connection

		//CONV回畳込み（+ReLU)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st; wsize *= st;
			if (s == 0) BBackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], lr, 1);
			//else if (s == SEG - 1) BBackGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dbias[s][c], dwoi[s][c], out[s][c + 1],
			//	gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], gdelta[s][c], dgbias[s][c], dgwoi[s][c], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], lr);
			/*else BBackCSGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c],
				dbias[s][c], dwoi[s][c], out[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], lr,
				cabias[s][c], cawoi[s][c], caout[s][c], cunitout[s][c], gapin[s][c], camean[s][c], cavar[s][c], cagamma[s][c], cabeta[s][c],
				sabias[s][c], sawoi[s][c], saout[s][c], sunitout[s][c], samean[s][c], savar[s][c], sagamma[s][c], sabeta[s][c],resbias[s][c], reswoi[s][c], resout[s][c],
				dcabias[s][c], dcawoi[s][c], dsabias[s][c], dsawoi[s][c], dresbias[s][c], dreswoi[s][c]);*/
			/*else BBackCGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c],
			dbias[s][c], dwoi[s][c], out[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
			bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], lr, 
			cabias[s][c], cawoi[s][c], caout[s][c], cunitout[s][c], gapin[s][c], camean[s][c], cavar[s][c], cagamma[s][c], cabeta[s][c], dcabias[s][c], dcawoi[s][c]);*/
			else BBackSGhostbottleneck(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c],
				dbias[s][c], dwoi[s][c], out[s][c + 1], gbias[s][c], gwoi[s][c], gin[s][c], gout[s][c], dgbias[s][c], dgwoi[s][c],
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], gbnet[s][c], gmean[s][c], gvar[s][c], ggamma[s][c], gbeta[s][c], lr, 
				sabias[s][c], sawoi[s][c], saout[s][c], sunitout[s][c], samean[s][c], savar[s][c], sagamma[s][c], sabeta[s][c], dsabias[s][c], dsawoi[s][c]);
		}
		if (s == 0) for (j = 0; j < chN[s][0]; j++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)//出力側
			delta[s][0][j][b][h][w] += skipdelta[s][j][b][h][w];//skip-connection
	}
	BInBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dibias, diwoi, out[0][0], bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], lr);

	// 重み更新 重み修正量を加算した回数で調整
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (bias[s][c][j] += -lr * dbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					woi[s][c][j][i][y][x] += -lr * dwoi[s][c][j][i][y][x] / (double)w;
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (gbias[s][c][j] += -lr * dgbias[s][c][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					gwoi[s][c][j][i][y][x] += -lr * dgwoi[s][c][j][i][y][x] / (double)w;

		//以下3つ提案
		for (w = 1 * 1 * BS, c = 0; c < CONV; c++) for (p = 0; p < 2; p++)for (j = 0; j < chN[s][c + 1]; j++)
			for (cabias[s][c][p][j] += -lr * dcabias[s][c][p][j] / (double)w, i = 0; i < chN[s][c + 1]; i++)
				cawoi[s][c][p][j][i] += -lr * dcawoi[s][c][p][j][i] / (double)w;
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (p = 0; p < 2; p++)for (j = 0; j < chN[s][c + 1]; j++)
			for (sabias[s][c][p][j] += -lr * dsabias[s][c][p][j] / (double)w, i = 0; i < chN[s][c + 1]; i++)
				sawoi[s][c][p][j][i] += -lr * dsawoi[s][c][p][j][i] / (double)w;
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) for (j = 0; j < chN[s][c + 1]; j++)
			for (resbias[s][c][j] += -lr * dresbias[s][c][j] / (double)w, i = 0; i < chN[s][c + 1] * 2; i++)
				reswoi[s][c][j][i] += -lr * dreswoi[s][c][j][i] / (double)w;

		w /= PS * PS;
		if (s == SEG - 1) och = CHMAX; else och = chN[s + 1][0];
		for (j = 0; j < och; j++)
			for (skipbias[s][j] += -lr * dskipbias[s][j] / (double)w, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipwoi[s][j][i][y][x] += -lr * dskipwoi[s][j][i][y][x] / (double)w;
	}
	w = (H - IKS + 1) * (W - IKS + 1) * BS;
	for (j = 0; j < chN[0][0]; j++)
		for (inbias[j] += -lr * dibias[j] / (double)w, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inwoi[j][i][y][x] += -lr * diwoi[j][i][y][x] / (double)w; //0
}
