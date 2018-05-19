#pragma once

#include <cv.h>
#include <cv.hpp>

using namespace cv;

Mat SourceImage;
Mat GlobalResult;

#define PI 3.1415926535897932
#define e 2.718281828459

void Erosion(Mat Source, Mat &Result, Mat &Kernel, Point &Origin)
{
	Result = Mat(Source.rows, Source.cols, Source.type());
	
	int KernelWidth = Kernel.cols, KernelHeight = Kernel.rows;

	bool **KernelMap = new bool*[KernelHeight];
	for (int i = 0; i < KernelHeight; ++i)
	{
		KernelMap[i] = new bool[KernelWidth];
	}

	for (int i = 0; i < Kernel.rows; ++i)
	{
		uchar *pKernel = Kernel.ptr<uchar>(i);
		for (int j = 0; j < Kernel.cols; ++j)
		{
			if (*pKernel)
				KernelMap[i][j] = true;
			else
				KernelMap[i][j] = false;
			++pKernel;
		}
	}

	Mat ExtensionSource;
	copyMakeBorder(Source, ExtensionSource, Origin.y, Kernel.rows - Origin.y - 1, Origin.x, Kernel.cols - Origin.x - 1, IPL_BORDER_REPLICATE);

	uchar **pExtensionSourceRow = new uchar*[ExtensionSource.rows];
	for (int i = 0; i < ExtensionSource.rows; ++i)
	{
		pExtensionSourceRow[i] = ExtensionSource.ptr<uchar>(i);
	}

	int Channel = Result.channels();

#pragma omp parallel for
	for (int i = 0; i < Result.rows; ++i)
	{
		uchar *pResult = Result.ptr<uchar>(i);
		int ExtensionRow = i + Origin.y;
		for (int j = 0; j < Result.cols; ++j)
		{
			int ExtensionCol = j + Origin.x;
			for (int c = 0; c < Channel; ++c)
			{
				int Min = pExtensionSourceRow[i][j * Channel + c];
				int Data;
				for (int m = 0; m < Kernel.rows; ++m)
				{
					for (int n = 0; n < Kernel.cols; ++n)
					{
						if (KernelMap[m][n])
						{
							Data = pExtensionSourceRow[i + m][(j + n) * Channel + c];
							if (Data < Min)
								Min = Data;
						}
					}
				}
				*pResult++ = Min;
			}
		}
	}

	delete[] pExtensionSourceRow;
	for (int i = 0; i < KernelHeight; ++i)
	{
		delete[] KernelMap[i];
	}
	delete[] KernelMap;
}

void Dilation(Mat Source, Mat &Result, Mat &Kernel, Point &Origin)
{
	Result = Mat(Source.rows, Source.cols, Source.type());

	int KernelWidth = Kernel.cols, KernelHeight = Kernel.rows;

	bool **KernelMap = new bool*[KernelHeight];
	for (int i = 0; i < KernelHeight; ++i)
	{
		KernelMap[i] = new bool[KernelWidth];
	}

	for (int i = 0; i < Kernel.rows; ++i)
	{
		uchar *pKernel = Kernel.ptr<uchar>(i);
		for (int j = 0; j < Kernel.cols; ++j)
		{
			if (*pKernel)
				KernelMap[i][j] = true;
			else
				KernelMap[i][j] = false;
			++pKernel;
		}
	}

	Mat ExtensionSource;
	copyMakeBorder(Source, ExtensionSource, Origin.y, Kernel.rows - Origin.y - 1, Origin.x, Kernel.cols - Origin.x - 1, IPL_BORDER_REPLICATE);

	uchar **pExtensionSourceRow = new uchar*[ExtensionSource.rows];
	for (int i = 0; i < ExtensionSource.rows; ++i)
	{
		pExtensionSourceRow[i] = ExtensionSource.ptr<uchar>(i);
	}

	int Channel = Result.channels();

#pragma omp parallel for
	for (int i = 0; i < Result.rows; ++i)
	{
		uchar *pResult = Result.ptr<uchar>(i);
		int ExtensionRow = i + Origin.y;
		for (int j = 0; j < Result.cols; ++j)
		{
			int ExtensionCol = j + Origin.x;
			for (int c = 0; c < Channel; ++c)
			{
				int Max = pExtensionSourceRow[i][j * Channel + c];
				int Data;
				for (int m = 0; m < Kernel.rows; ++m)
				{
					for (int n = 0; n < Kernel.cols; ++n)
					{
						if (KernelMap[m][n])
						{
							Data = pExtensionSourceRow[i + m][(j + n) * Channel + c];
							if (Data > Max)
								Max = Data;
						}
					}
				}
				*pResult++ = Max;
			}
		}
	}

	delete[] pExtensionSourceRow;
	for (int i = 0; i < KernelHeight; ++i)
	{
		delete[] KernelMap[i];
	}
	delete[] KernelMap;
}