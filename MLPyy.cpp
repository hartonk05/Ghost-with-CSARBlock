#include<math.h>
#include"MLPyy.h"

int Softmax = 1;
int BatchNormal = 1;

void Forward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]) {
	int i, j, l;
	double sum;
	for (l = 0; l < L - 2; l++)//一番出力側の中間層まで
		for (j = 0; j < nodeN[l + 1]; j++) {//出力側
			for (sum = bias[l][j], i = 0; i < nodeN[l]; i++) //入力側
				sum += woi[l][j][i] * out[l][i];
			//バッチ学習時の補正
			if (BATCHLEARN && BatchNormal) sum = (sum - mean[l + 1][j])/sqrt(var[l + 1][j]+EPS) * gamma[l + 1][j] + beta[l + 1][j];
			if (RELU) {
				if (sum > 0) out[l + 1][j] = sum;
				else out[l + 1][j] = 0;
			}
			else out[l + 1][j] = 1.0 / (1.0 + exp(-sum)); //sigmoid
		}
	// 出力層の計算
	for (j = 0; j < nodeN[l + 1]; j++) {//出力側
		for (out[l + 1][j] = bias[l][j], i = 0; i < nodeN[l]; i++) //入力側
			out[l + 1][j] += woi[l][j][i] * out[l][i];
	}
	if (Softmax) {
		for (sum = 0, j = 0; j < nodeN[l + 1]; j++) sum += exp(out[l + 1][j]);
		for (j = 0; j < nodeN[l + 1]; j++) out[l + 1][j] = exp(out[l + 1][j]) / sum;
	}
	else for (j = 0; j < nodeN[l + 1]; j++)
		out[l + 1][j] = 1.0 / (1.0 + exp(-out[l + 1][j])); //sigmoid
}
void BForward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX],
	double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double mmean[][NMAX], double mvar[][NMAX]) {
	int i, j, l, b;
	double sum, net[BS];

	for (l = 0; l < L - 2; l++) {//一番出力側の中間層まで	
		for (j = 0; j < nodeN[l + 1]; j++) {//出力側
			for (b = 0; b < BS; b++)for (net[b] = bias[l][j], i = 0; i < nodeN[l]; i++) //入力側
				net[b] += woi[l][j][i] * out[l][b][i];//入力の総和を計算
			// Batch正則化
			if (BatchNormal) {
				for (b = 0; b < BS; b++)bnet[l + 1][j][b] = net[b];
				BatchNormalF(bnet[l + 1][j], net, &mean[l + 1][j], &var[l + 1][j], gamma[l + 1][j], beta[l + 1][j]);
				mmean[l + 1][j] = mmean[l + 1][j] * MRATE + (1.0 - MRATE) * mean[l + 1][j];
				mvar[l + 1][j] = mvar[l + 1][j] * MRATE + (1.0 - MRATE) * var[l + 1][j];
			}
			for (b = 0; b < BS; b++)//出力値の計算
				if (RELU) {
					if (net[b] > 0) out[l + 1][b][j] = net[b];
					else out[l + 1][b][j] = 0;
				}
				else out[l + 1][b][j] = 1.0 / (1.0 + exp(-net[b])); //sigmoid
		}
	}
	// 出力層の計算
	for (b = 0; b < BS; b++) {
		for (j = 0; j < nodeN[l + 1]; j++) {//出力側
			for (out[l + 1][b][j] = bias[l][j], i = 0; i < nodeN[l]; i++) //入力側
				out[l + 1][b][j] += woi[l][j][i] * out[l][b][i];
		}
		if (Softmax) {
			for (sum = 0, j = 0; j < nodeN[l + 1]; j++) sum += exp(out[l + 1][b][j]);
			for (j = 0; j < nodeN[l + 1]; j++) out[l + 1][b][j] = exp(out[l + 1][b][j]) / sum;
		}
		else for (j = 0; j < nodeN[l + 1]; j++)
			out[l + 1][b][j] = 1.0 / (1.0 + exp(-out[l + 1][b][j])); //sigmoid
	}
}

void BackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double tk[], double delta[][NMAX], double lr) {
	int i, j, l, k;
	double sum;

	if (Softmax) for (l = L - 1, k = 0; k < nodeN[l]; k++)
		delta[l][k] = (out[l][k] - tk[k]); //softmax+CrossEntropyの微分
	else for (l = L - 1, k = 0; k < nodeN[l]; k++)
		delta[l][k] = (out[l][k] - tk[k]) * (1.0 - out[l][k]) * out[l][k]; //sigmoid+二乗誤差

	for (l = L - 2; l > 0; l--)
		for (j = 0; j < nodeN[l]; j++) {//入力側
			for (sum = 0, k = 0; k < nodeN[l + 1]; k++)//出力側 
				sum += delta[l + 1][k] * woi[l][k][j];
			if (RELU) {
				if (out[l][j] > 0) delta[l][j] = sum;
				else delta[l][j] = 0;
			}
			else delta[l][j] = sum * (1.0 - out[l][j]) * out[l][j];//sigmoid
		}
	if (NETMODEL !=0)//deltaを０層まで作る
		for (j = 0; j < nodeN[l]; j++) {//入力側
			for (sum = 0, k = 0; k < nodeN[l + 1]; k++)//出力側 
				sum += delta[l + 1][k] * woi[l][k][j];
			//↓CNNの最後の層の出力関数による
			if (out[l][j] > 0) delta[l][j] = sum;
			else delta[l][j] = 0;
		}

	// 重み更新
	for (l = 0; l < L - 1; l++)
		for (j = 0; j < nodeN[l + 1]; j++)//出力側
			for (bias[l][j] += -lr * delta[l + 1][j], i = 0; i < nodeN[l]; i++) //入力側
				woi[l][j][i] += -lr * delta[l + 1][j] * out[l][i];
}
void BBackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX],
	double tk[][K], double delta[][NMAX][BS], double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double lr) {
	int i, j, l, k, b;
	double sum;
	l = L - 1;
	for (b = 0; b < BS; b++) { // 出力層側のδ計算
		if (Softmax) for (k = 0; k < nodeN[l]; k++)
			delta[l][k][b] = (out[l][b][k] - tk[b][k]); //softmax+CrossEntropyの微分
		else for (k = 0; k < nodeN[l]; k++)
			delta[l][k][b] = (out[l][b][k] - tk[b][k]) * (1.0 - out[l][b][k]) * out[l][b][k]; //sigmoid+二乗誤差
	}
	for (l = L - 2; l > 0; l--) // 隠れ層のδ計算
		for (j = 0; j < nodeN[l]; j++) {//入力側
			for (b = 0; b < BS; b++) {
				for (sum = 0, k = 0; k < nodeN[l + 1]; k++)//出力側 
					sum += delta[l + 1][k][b] * woi[l][k][j];
				if (RELU) {
					if (out[l][b][j] > 0) delta[l][j][b] = sum;
					else delta[l][j][b] = 0;
				}
				else delta[l][j][b] = sum * (1.0 - out[l][b][j]) * out[l][b][j];//sigmoid
			}
			if (BatchNormal)BatchNormalB(delta[l][j], bnet[l][j], mean[l][j], var[l][j], &gamma[l][j], &beta[l][j], lr);
			//for (b = 0; b < BS; b++) {
			//	for (delta[l][j][b] = 0, k = 0; k < nodeN[l + 1]; k++)//出力側 
			//		delta[l][j][b] += delta[l + 1][k][b] * woi[l][k][j];
			//}
			//if (BatchNormal)BatchNormalB(delta[l][j], bnet[l][j], mean[l][j], var[l][j], &gamma[l][j], &beta[l][j]);
			//for (b = 0; b < BS; b++) {
			//	if (RELU) {
			//		if (out[l][b][j] == 0) delta[l][j][b] = 0;
			//	}
			//	else delta[l][j][b] = sum * (1.0 - out[l][b][j]) * out[l][b][j];//sigmoid
			//}
		}
	if (NETMODEL == 1||NETMODEL == 2 || NETMODEL == 3 || NETMODEL == 4)//deltaを０層まで作る
		for (j = 0; j < nodeN[l]; j++) {//入力側
			for (b = 0; b < BS; b++) {
				for (sum = 0, k = 0; k < nodeN[l + 1]; k++)//出力側 
					sum += delta[l + 1][k][b] * woi[l][k][j];
				//↓CNNの最後の層の出力関数による　基本ReLU
				if (out[l][b][j] > 0) delta[l][j][b] = sum;
				else delta[l][j][b] = 0;
			}
		}

	// 重み更新 batch数分蓄積させて合計/BSで１回だけ更新
	for (l = 0; l < L - 1; l++)
		for (j = 0; j < nodeN[l + 1]; j++) {//出力側
			for (sum = 0, b = 0; b < BS; b++) sum += delta[l + 1][j][b];
			bias[l][j] += -lr * sum / BS;
			for (i = 0; i < nodeN[l]; i++) {//入力側
				for (sum = 0, b = 0; b < BS; b++) sum += delta[l + 1][j][b] * out[l][b][i];
				woi[l][j][i] += -lr * sum / BS;
			}
		}
}

void BatchNormalF(double bnet[], double anet[], double* mean, double* var, double gamma, double beta) {
	double ave, sum, va, sqvar;
	int b;
	for (sum = 0, b = 0; b < BS; b++) sum += bnet[b];
	ave = sum / BS;
	for (sum = 0, b = 0; b < BS; b++) sum += (ave - bnet[b]) * (ave - bnet[b]);
	va = sum / BS;
	for (sqvar = sqrt(va + EPS), b = 0; b < BS; b++) anet[b] = gamma * (bnet[b] - ave) / sqvar + beta;
	*mean = ave;
	*var = va;
}

void BatchNormalB(double delta[], double bnet[], double mean, double var, double* gamma, double* beta, double lr) {
	int b;
	double f7= 1.0 / sqrt(var + EPS), tave, d3a[BS], sum, db, dg, ga= *gamma;

	double f3[BS], d12a[BS];
	for (b = 0; b < BS; b++)f3[b] = (bnet[b] - mean); //f3 xi-μ
	for (b = 0; b < BS; b++)d12a[b] = delta[b] * ga; //d12a
	for (tave = 0, b = 0; b < BS; b++)tave += f3[b] * d12a[b];
	tave /= BS;
	for (sum = 0, b = 0; b < BS; b++) {
		d3a[b] = f7 * (d12a[b] - f3[b] / (var + EPS) * tave); //d3a
		sum += d3a[b];
	}
	sum /= BS;
	for (db = 0, dg = 0, b = 0; b < BS; b++) {
		db += delta[b];
		dg += f3[b] * f7 * delta[b];
		delta[b] = d3a[b] - sum; //dx Delta (doutが更新される)
	}

	//for (tave = 0, b = 0; b < BS; b++)tave += (bnet[b] - mean) * delta[b] * ga;
	//tave /= BS;
	//for (sum = 0, b = 0; b < BS; b++) {
	//	d3a[b] = f7 * (delta[b] * ga - (bnet[b] - mean) / (var + EPS) * tave); //d3a
	//	sum += d3a[b];
	//}
	//sum /= BS;
	//for (db = 0, dg = 0, b = 0; b < BS; b++) {
	//	db += delta[b];
	//	dg += (bnet[b] - mean) * f7 * delta[b];
	//	delta[b] = d3a[b] - sum; //dx Delta (doutが更新される)
	//}
	*gamma += (-lr * dg / BS);
	*beta += (-lr * db / BS);
}