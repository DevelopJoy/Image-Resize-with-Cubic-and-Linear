//双线性插值
__global__ void zoomOutIn(const int n, const float*src, int srcWidth, int srcHeight, \
	float *dst, int dstWidth, int dstHeight) {

	float srcColTidf;
	float srcRowTidf;
	float c, r;
	const float rowScale = srcHeight / (float)(dstHeight);
	const float colScale = srcWidth / (float)(dstWidth);
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x) {
		int tidC = i;
		int tidR = i;// *colScaleExtend;
		float srcColTidf = (float)((tidC % (dstWidth)) * colScale);
		float srcRowTidf = (float)((tidR / (dstWidth)) * rowScale);
		int srcColTid = (int)srcColTidf;
		int srcRowTid = (int)srcRowTidf;
		c = srcColTidf - srcColTid;
		r = srcRowTidf - srcRowTid;

		int dstInd = i;
		int srcInd = srcRowTid * srcWidth + srcColTid;
		dst[dstInd] = 0;
		dst[dstInd] += (1 - c)*(1 - r)*src[srcRowTid * srcWidth + srcColTid];
		dst[dstInd] += (1 - c)*r*src[(srcRowTid + 1)*srcWidth + srcColTid];
		dst[dstInd] += c*(1 - r)*src[srcRowTid*srcWidth + srcColTid + 1];
		dst[dstInd] += c*r*src[(srcRowTid + 1)*srcWidth + srcColTid + 1];
	}
}

//双三次插值
__global__ void zoomCubicOutIn(const int n, const float*src, int srcWidth, int srcHeight, \
	float *dst, int dstWidth, int dstHeight) {

	float srcColTidf;
	float srcRowTidf;
	float c, r;
	float A = -0.75;
	const float rowScale = srcHeight / (float)(dstHeight);
	const float colScale = srcWidth / (float)(dstWidth);
	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x) {
		int tidC = i;
		int tidR = i;// *colScaleExtend;
		float srcColTidf = (float)((tidC % (dstWidth)) * colScale);
		float srcRowTidf = (float)((tidR / (dstWidth)) * rowScale);
		int srcColTid = (int)srcColTidf;
		int srcRowTid = (int)srcRowTidf;
		c = srcColTidf - srcColTid;
		r = srcRowTidf - srcRowTid;

		int dstInd = i;
		int srcInd = srcRowTid * srcWidth + srcColTid;
		dst[dstInd] = 0;

		{
			//
			float coeffsY[4];
			coeffsY[0] = ((A*(r + 1) - 5 * A)*(r + 1) + 8 * A)*(r + 1) - 4 * A;
			coeffsY[1] = ((A + 2)*r - (A + 3))*r*r + 1;
			coeffsY[2] = ((A + 2)*(1 - r) - (A + 3))*(1 - r)*(1 - r) + 1;
			coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

			float coeffsX[4];
			coeffsX[0] = ((A*(c + 1) - 5 * A)*(c + 1) + 8 * A)*(c + 1) - 4 * A;
			coeffsX[1] = ((A + 2)*c - (A + 3))*c*c + 1;
			coeffsX[2] = ((A + 2)*(1 - c) - (A + 3))*(1 - c)*(1 - c) + 1;
			coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

			dst[dstInd] =
				src[(srcRowTid - 1) * srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid - 1)] * coeffsX[0] * coeffsY[3] \
				+ src[(srcRowTid - 1) * srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid)] * coeffsX[1] * coeffsY[3] \
				+ src[(srcRowTid - 1) * srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid + 1)] * coeffsX[2] * coeffsY[3] \
				+ src[(srcRowTid - 1) * srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[0] \
				+ src[(srcRowTid)* srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[1] \
				+ src[(srcRowTid + 1) * srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[2] \
				+ src[(srcRowTid + 2) * srcWidth + (srcColTid + 2)] * coeffsX[3] * coeffsY[3];
		}
	}
}
