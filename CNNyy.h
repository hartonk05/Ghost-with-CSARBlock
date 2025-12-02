#pragma once
#define SEG 3
#define CONV 2
//#define BOTTLENECK 2
#define CHMAX 32 //MNIST：64,cifarきついたぶん
#define S 4 //分割数

#define KS 3 //CNNのカーネルサイズ
#define PS 2 //pooling size

void CForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS],
	double out[][CONV + 1][CHMAX][H][W], double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX]);
void BCForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS],
	double out[][CONV + 1][CHMAX][BS][H][W], double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W],
	double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX]);

void SkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]);
void BSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]);

void GForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double mout[],
	double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][H][W], double gout[SEG][CONV][CHMAX][H][W],
	double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX]);
void BGForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double mout[][NMAX],
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], //
	double gbnet[SEG][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX]);

void CSGForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][H][W], double gout[SEG][CONV][CHMAX][H][W], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][H][W], double cagamma[][CONV][2][CHMAX], double cabeta[][CONV][2][CHMAX], double cammean[][CONV][2][CHMAX], double camvar[][CONV][2][CHMAX], double cunitout[SEG][CONV][CHMAX][H][W], double gapin[SEG][CONV][CHMAX][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][H][W], double sagamma[][CONV][2][CHMAX], double sabeta[][CONV][2][CHMAX], double sammean[][CONV][2][CHMAX], double samvar[][CONV][2][CHMAX], double sunitout[SEG][CONV][CHMAX][H][W],
	double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][H][W]);

void BCSGForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], double gbnet[SEG][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX], double gmmean[][CONV][CHMAX], double gmvar[][CONV][CHMAX],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][BS][H][W], double cagamma[][CONV][2][CHMAX], double cabeta[][CONV][2][CHMAX], double cammean[][CONV][2][CHMAX], double camvar[][CONV][2][CHMAX], double cunitout[SEG][CONV][CHMAX][BS][H][W], double gapin[SEG][CONV][CHMAX][BS][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][BS][H][W], double sagamma[][CONV][2][CHMAX], double sabeta[][CONV][2][CHMAX], double sammean[][CONV][2][CHMAX], double samvar[][CONV][2][CHMAX], double sunitout[SEG][CONV][CHMAX][BS][H][W],
	double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][BS][H][W]);

void CBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr);
void BCBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr,
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX]);
void SkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, double in[ICH][H][W],
	double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][H][W], double skipdelta[SEG][CHMAX][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS]);
void BSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[NMAX][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr,
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][BS][H][W], double skipdelta[SEG][CHMAX][BS][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX],
	double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS]);

void GBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, double in[ICH][H][W],
	double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][H][W], double skipdelta[SEG][CHMAX][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[][CONV][CHMAX][H][W], double gout[][CONV][CHMAX][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS]);
void BGBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[NMAX][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr,
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][BS][H][W], double skipdelta[SEG][CHMAX][BS][H][W],
	double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS],
	double gbnet[][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX]);

void CSGBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W], double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr, double in[ICH][H][W],
	double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][H][W], double skipdelta[SEG][CHMAX][H][W], double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[][CONV][CHMAX][H][W], double gout[][CONV][CHMAX][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][H][W], double cunitout[SEG][CONV][CHMAX][H][W], double gapin[SEG][CONV][CHMAX][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][H][W], double sunitout[SEG][CONV][CHMAX][H][W], double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][H][W],
	double dcabias[][CONV][2][CHMAX], double dcawoi[][CONV][2][CHMAX][CHMAX], double dsabias[][CONV][2][CHMAX], double dsawoi[][CONV][2][CHMAX][CHMAX], double dresbias[][CONV][CHMAX], double dreswoi[][CONV][CHMAX][CHMAX * 2]);
void BCSGBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W], double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[NMAX][BS], double dbias[][CONV][CHMAX], double dwoi[][CONV][CHMAX][CHMAX][KS][KS], double lr,
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inbias[], double inwoi[CHMAX][ICH][IKS][IKS], double skipout[SEG][CHMAX][BS][H][W], double skipdelta[SEG][CHMAX][BS][H][W], double skipbias[SEG][CHMAX], double skipwoi[SEG][CHMAX][CHMAX][KS][KS], double dskipbias[SEG][CHMAX], double dskipwoi[SEG][CHMAX][CHMAX][KS][KS], double dibias[CHMAX], double diwoi[CHMAX][ICH][IKS][IKS],
	double gin[SEG][CONV][CHMAX][BS][H][W], double gout[SEG][CONV][CHMAX][BS][H][W], double gbias[][CONV][CHMAX], double gwoi[][CONV][CHMAX][CHMAX][KS][KS], double dgbias[][CONV][CHMAX], double dgwoi[][CONV][CHMAX][CHMAX][KS][KS],
	double gbnet[][CONV][CHMAX][BS][H][W], double gmean[][CONV][CHMAX], double gvar[][CONV][CHMAX], double ggamma[][CONV][CHMAX], double gbeta[][CONV][CHMAX],
	double cabias[][CONV][2][CHMAX], double cawoi[][CONV][2][CHMAX][CHMAX], double caout[SEG][CONV][2][CHMAX][BS][H][W], double cunitout[SEG][CONV][CHMAX][BS][H][W], double gapin[SEG][CONV][CHMAX][BS][H][W],
	double sabias[][CONV][2][CHMAX], double sawoi[][CONV][2][CHMAX][CHMAX], double saout[SEG][CONV][2][CHMAX][BS][H][W], double sunitout[SEG][CONV][CHMAX][BS][H][W], double resbias[][CONV][CHMAX], double reswoi[][CONV][CHMAX][CHMAX * 2], double resout[SEG][CONV][CHMAX][BS][H][W],
	double camean[][CONV][2][CHMAX], double cavar[][CONV][2][CHMAX], double cagamma[][CONV][2][CHMAX], double cabeta[][CONV][2][CHMAX], double samean[][CONV][2][CHMAX], double savar[][CONV][2][CHMAX], double sagamma[][CONV][2][CHMAX], double sabeta[][CONV][2][CHMAX],
	double dcabias[][CONV][2][CHMAX], double dcawoi[][CONV][2][CHMAX][CHMAX], double dsabias[][CONV][2][CHMAX], double dsawoi[][CONV][2][CHMAX][CHMAX], double dresbias[][CONV][CHMAX], double dreswoi[][CONV][CHMAX][CHMAX * 2]);
