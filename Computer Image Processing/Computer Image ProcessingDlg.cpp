
// Computer Image ProcessingDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "Computer Image Processing.h"
#include "Computer Image ProcessingDlg.h"
#include "afxdialogex.h"
#include <iostream>
#include <cv.h>
#include <cv.hpp>
#include <omp.h>
#include "ProcessFunction.h"
#include "CDialog_1.h"

using namespace std;
using namespace cv;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
public:
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CComputerImageProcessingDlg 对话框



CComputerImageProcessingDlg::CComputerImageProcessingDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_COMPUTERIMAGEPROCESSING_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CComputerImageProcessingDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CComputerImageProcessingDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_COMMAND(ID_32771, &CComputerImageProcessingDlg::OnOpenImageFile)
	ON_COMMAND(ID_32772, &CComputerImageProcessingDlg::OnAllRed)
	ON_COMMAND(ID_32773, &CComputerImageProcessingDlg::OnRedBlue)
	ON_COMMAND(ID_32774, &CComputerImageProcessingDlg::OnGray)
	ON_COMMAND(ID_32776, &CComputerImageProcessingDlg::OnImageZoom_Normal)
	ON_COMMAND(ID_32777, &CComputerImageProcessingDlg::OnImageZoom_Linear)
	ON_COMMAND(ID_32778, &CComputerImageProcessingDlg::OnImageZoom_Cubic)
	ON_COMMAND(ID_32779, &CComputerImageProcessingDlg::OnBinaryzation_GivenThreshold)
	ON_COMMAND(ID_32780, &CComputerImageProcessingDlg::OnHistogramEqualization)
	ON_COMMAND(ID_32781, &CComputerImageProcessingDlg::OnInvert)
	ON_COMMAND(ID_32782, &CComputerImageProcessingDlg::OnCustomHistogramMapping)
	ON_COMMAND(ID_32783, &CComputerImageProcessingDlg::OnBitPlaneSlicing)
	ON_COMMAND(ID_32784, &CComputerImageProcessingDlg::OnFastGaussianFilter)
END_MESSAGE_MAP()


// CComputerImageProcessingDlg 消息处理程序

BOOL CComputerImageProcessingDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CComputerImageProcessingDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CComputerImageProcessingDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CComputerImageProcessingDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CComputerImageProcessingDlg::OnOpenImageFile()
{
	string FilePathName;
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, (LPCTSTR)_TEXT("Image Files|*.jpg;*.bmp;*.png;*.jpeg|All Files (*.*)|*.*||"), NULL);
	if (dlg.DoModal() == IDOK)
	{
		FilePathName = dlg.GetPathName();
	}
	else
	{
		return;
	}

	SourceImage = imread(FilePathName);
	imshow("Source", SourceImage);
}



void CComputerImageProcessingDlg::OnAllRed()
{
	Mat AllRed(Size(800, 600), CV_8UC3, Scalar(0, 0, 255));

	imshow("Result", AllRed);
}

void CComputerImageProcessingDlg::OnRedBlue()
{
	Mat ResultImage(Size(800, 600), CV_8UC3);

	bool Color = true;

	for (int i = 0; i < ResultImage.cols / 50; ++i)
	{
		for (int j = 0; j < ResultImage.rows / 50; ++j)
		{
			for (int m = 0; m < 50; ++m)
			{
				for (int n = 0; n < 50; ++n)
				{
					if (Color)
					{
						ResultImage.at<Vec3b>(j * 50 + m, i * 50 + n)[0] = 255;
						ResultImage.at<Vec3b>(j * 50 + m, i * 50 + n)[1] = 0;
						ResultImage.at<Vec3b>(j * 50 + m, i * 50 + n)[2] = 0;
					}
					else
					{
						ResultImage.at<Vec3b>(j * 50 + m, i * 50 + n)[0] = 0;
						ResultImage.at<Vec3b>(j * 50 + m, i * 50 + n)[1] = 0;
						ResultImage.at<Vec3b>(j * 50 + m, i * 50 + n)[2] = 255;
					}
				}
			}
			Color = !Color;
		}
		Color = !Color;
	}

	imshow("Result", ResultImage);
}

void CComputerImageProcessingDlg::OnGray()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 3)
	{
		AfxMessageBox(_T("请使用三通道彩色图片！"));
		return;
	}
	Mat Gray;

	Gray.create(SourceImage.size(), SourceImage.type());

	int i, j;
	uchar *pResult, *pSource;

	for (i = 0; i < SourceImage.rows; i++)
	{
		pResult = Gray.ptr<uchar>(i);
		pSource = SourceImage.ptr<uchar>(i);
		for (j = 0; j < SourceImage.cols; ++j)
		{
			int g = 0.114 * pSource[0] + 0.587 * pSource[1] + 0.299 * pSource[2];
			*pResult++ = g;
			*pResult++ = g;
			*pResult++ = g;
			pSource += 3;
		}
	}
	imshow("Result", Gray);
}

void CComputerImageProcessingDlg::OnBinaryzation_GivenThreshold()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	static int Threshold;
	Threshold = 122;

	cvtColor(SourceImage, SourceImage, CV_RGB2GRAY);
	imshow("Source", SourceImage);

	namedWindow("二值化(给定阈值)");

	struct
	{
		static void Binaryzation(int Pos, void* ext)
		{
			Mat Result(SourceImage.rows, SourceImage.cols, CV_8U);
			int i, j;
			uchar *pResult, *pSource;

			for (i = 0; i < Result.rows; i++)
			{
				pResult = Result.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(i);
				for (j = 0; j < Result.cols; ++j)
				{
					if (*pSource <= Pos)
						*pResult = 0;
					else
						*pResult = 255;
					++pResult;
					++pSource;
				}
			}
			imshow("二值化(给定阈值)", Result);
		}
	}Binaryzation;

	Binaryzation.Binaryzation(Threshold, NULL);
	createTrackbar("阈值", "二值化(给定阈值)", &Threshold, 255, Binaryzation.Binaryzation);
}

void CComputerImageProcessingDlg::OnImageZoom_Normal()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	static int RowSlider, ColSlider;
	RowSlider = 50;
	ColSlider = 50;

	namedWindow("最邻近插值"); 
	imshow("最邻近插值", SourceImage);

	struct
	{
		static void Zoom(int Pos, void* ext)
		{
			double RowScaling, ColScaling;
			RowScaling = (RowSlider - 50) / 100.0 + 1;
			ColScaling = (ColSlider - 50) / 100.0 + 1;
			Mat Result(RowScaling * SourceImage.rows, ColScaling * SourceImage.cols, SourceImage.type());
			int i, j;
			uchar *pResult, *pSource;
			int RowInSource;

			int* MapCol = new int[Result.cols];
			for (i = 0; i < Result.cols; ++i)
			{
				MapCol[i] = (int)(i / ColScaling) * 3;
			}

			for (i = 0; i < Result.rows; i++)
			{
				RowInSource = i / RowScaling;
				pResult = Result.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(RowInSource);
				for (j = 0; j < Result.cols; ++j)
				{
					*pResult++ = pSource[MapCol[j]];
					*pResult++ = pSource[MapCol[j] + 1];
					*pResult++ = pSource[MapCol[j] + 2];
				}
			}
			delete[] MapCol;
			imshow("最邻近插值", Result);
		}
	}Zoom_Normal;

	createTrackbar("宽度缩放", "最邻近插值", &ColSlider, 100, Zoom_Normal.Zoom);
	createTrackbar("高度缩放", "最邻近插值", &RowSlider, 100, Zoom_Normal.Zoom);
}

void CComputerImageProcessingDlg::OnImageZoom_Linear()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	static int RowSlider, ColSlider;
	RowSlider = 50;
	ColSlider = 50;

	namedWindow("双线性插值");
	imshow("双线性插值", SourceImage);

	struct
	{
		static void Zoom(int Pos, void* ext)
		{
			float RowScaling, ColScaling;
			RowScaling = (RowSlider - 50) / 100.0 + 1;
			ColScaling = (ColSlider - 50) / 100.0 + 1;
			Mat Result(RowScaling * SourceImage.rows, ColScaling * SourceImage.cols, SourceImage.type());
			int i, j;

			struct MappingTable
			{
				unsigned short left, right;
				float Proportion;
			}*MapCol;

			MapCol = new MappingTable[Result.cols];

			float MappingPos;
			for (i = 0; i < Result.cols; ++i)
			{
				MappingPos = i / ColScaling;
				MapCol[i].Proportion = (MappingPos - (int)MappingPos) / 1.0;
				MapCol[i].left = (int)MappingPos * 3;
				MapCol[i].right = MapCol[i].left + 3;
			}

			uchar *pResult, *pSource;
			float RowInSource, RowProportion;
			unsigned int top, bottom;
			for (i = 0; i < Result.rows-1; i++)
			{
				RowInSource = i / RowScaling;
				top = RowInSource;
				bottom = RowInSource + 1;
				
				if (bottom == SourceImage.rows)
					bottom = SourceImage.rows - 1;
				
				RowProportion = (RowInSource - top) / 1.0;

				pResult = Result.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(top);
				for (j = 0; j < Result.cols; ++j)
				{
					*pResult++ = pSource[MapCol[j].left] + (pSource[MapCol[j].right] - pSource[MapCol[j].left]) * MapCol[j].Proportion;
					*pResult++ = pSource[MapCol[j].left + 1] + (pSource[MapCol[j].right + 1] - pSource[MapCol[j].left + 1]) * MapCol[j].Proportion;
					*pResult++ = pSource[MapCol[j].left + 2] + (pSource[MapCol[j].right + 2] - pSource[MapCol[j].left + 2]) * MapCol[j].Proportion;
				}

				pResult = Result.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(bottom);
				for (j = 0; j < Result.cols; ++j)
				{
					*pResult++ = (pSource[MapCol[j].left] + (pSource[MapCol[j].right] - pSource[MapCol[j].left]) * MapCol[j].Proportion - (*pResult)) * RowProportion + (*pResult);
					*pResult++ = (pSource[MapCol[j].left + 1] + (pSource[MapCol[j].right + 1] - pSource[MapCol[j].left + 1]) * MapCol[j].Proportion - (*pResult)) * RowProportion + (*pResult);
					*pResult++ = (pSource[MapCol[j].left + 2] + (pSource[MapCol[j].right + 2] - pSource[MapCol[j].left + 2]) * MapCol[j].Proportion - (*pResult)) * RowProportion + (*pResult);
				}
			}
			delete[] MapCol;

			imshow("双线性插值", Result);
		}
	}Zoom_Linear;

	createTrackbar("宽度缩放", "双线性插值", &ColSlider, 100, Zoom_Linear.Zoom);
	createTrackbar("高度缩放", "双线性插值", &RowSlider, 100, Zoom_Linear.Zoom);
}

void CComputerImageProcessingDlg::OnImageZoom_Cubic()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	static int RowSlider, ColSlider;
	RowSlider = 50;
	ColSlider = 50;

	namedWindow("双三次插值");
	imshow("双三次插值", SourceImage);

	struct
	{
		static void Zoom(int Pos, void* ext)
		{
			double RowScaling, ColScaling;
			RowScaling = (RowSlider - 50) / 100.0 + 1;
			ColScaling = (ColSlider - 50) / 100.0 + 1;
			Mat Result;
			resize(SourceImage, Result, Size(ColScaling * SourceImage.cols, RowScaling * SourceImage.rows), INTER_CUBIC);

			imshow("双三次插值", Result);
		}
	}Zoom_Cubic;

	createTrackbar("宽度缩放", "双三次插值", &ColSlider, 100, Zoom_Cubic.Zoom);
	createTrackbar("高度缩放", "双三次插值", &RowSlider, 100, Zoom_Cubic.Zoom);

	AfxMessageBox("This function is powered by OpenCV");
}

void CComputerImageProcessingDlg::OnHistogramEqualization()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if(SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, CV_RGB2GRAY);

	imshow("Source", SourceImage);

	Mat Result(SourceImage.rows,SourceImage.cols, SourceImage.type());

	int PixelCount[256];
	float PixelRate[256];
	memset(PixelCount, 0, sizeof(PixelCount));
	memset(PixelRate, 0, sizeof(PixelRate));

	int i, j;

	uchar *pSource, *pResult;
	for (i = 0; i < SourceImage.rows; i++)
	{
		pSource = SourceImage.ptr<uchar>(i);
		for (j = 0; j < SourceImage.cols; ++j)
		{
			++PixelCount[*pSource++];
		}
	}

	int PixelNumber = SourceImage.rows * SourceImage.cols;
	PixelRate[0] = (double)PixelCount[0] / PixelNumber;
	for (i = 1; i < 256; ++i)
	{
		PixelRate[i] = (double)PixelCount[i] / PixelNumber + PixelRate[i-1];
	}

	for (i = 0; i < 256; ++i)
	{
		PixelCount[i] = (PixelRate[i] * 256 - 0.5);
	}

	for (i = 0; i < Result.rows; i++)
	{
		pResult = Result.ptr<uchar>(i);
		pSource = SourceImage.ptr<uchar>(i);
		for (j = 0; j < SourceImage.cols; ++j)
		{
			*pResult++ = PixelCount[*pSource++];
		}
	}

	imshow("直方图均衡化（灰度）", Result);
}

void CComputerImageProcessingDlg::OnInvert()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

	int ColorMap[256];
	int i, j;
	for (i = 0; i < 256; ++i)
		ColorMap[i] = 255 - i;
	uchar *pSource, *pResult;
	for (i = 0; i < Result.rows; i++)
	{
		pResult = Result.ptr<uchar>(i);
		pSource = SourceImage.ptr<uchar>(i);
		for (j = 0; j < SourceImage.cols * SourceImage.channels(); ++j)
		{
			*pResult++ = ColorMap[*pSource++];
		}
	}

	imshow("反相", Result);
}

void CComputerImageProcessingDlg::OnCustomHistogramMapping()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, CV_RGB2GRAY);
	imshow("Source", SourceImage);

	CDialog_1 Dialog;
	Dialog.DoModal();

	int Mapping[256];
	int i, j;

	for (i = 0; i < 256; ++i)
	{
		Mapping[i] = Dialog.MappingTable[i];
	}

	Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());
	
	uchar *pSource, *pResult;
	for (i = 0; i < Result.rows; i++)
	{
		pResult = Result.ptr<uchar>(i);
		pSource = SourceImage.ptr<uchar>(i);
		for (j = 0; j < SourceImage.cols; ++j)
		{
			*pResult++ = Mapping[*pSource++];
		}
	}

	imshow("自定义函数直方图均衡化（灰度）", Result);
}

void CComputerImageProcessingDlg::OnBitPlaneSlicing()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, CV_RGB2GRAY);

	imshow("Source", SourceImage);
	namedWindow("比特平面分层");

	static Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

	static int BitPlane;
	BitPlane = 0;

	struct
	{
		static void Slicing(int Pos, void* ext)
		{
			int i, j;

			uchar *pSource, *pResult;
			for (i = 0; i < Result.rows; i++)
			{
				pResult = Result.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(i);
				for (j = 0; j < SourceImage.cols; ++j)
				{
					*pResult = *pSource++ & (1 << BitPlane);
					if (*pResult != 0)
						*pResult = 255;
					++pResult;
				}
			}

			imshow("比特平面分层", Result);
		}
	}BitPlaneSlicing;

	BitPlaneSlicing.Slicing(0, 0);

	createTrackbar("比特平面选择", "比特平面分层", &BitPlane, 7, BitPlaneSlicing.Slicing);
}


void CComputerImageProcessingDlg::OnFastGaussianFilter()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, CV_RGB2GRAY);

	imshow("Source", SourceImage);
	namedWindow("快速高斯滤波");

	static Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

	static int Radius, Variance;
	Radius = 3;
	Variance = 1;

	struct
	{
		static void Filter(int Pos, void* ext)
		{
			double *GaussianTable = new double[Radius + 1];
			int *GaussianTable_Int = new int[Radius + 1];
			int i, j;

			double xRatio = (double)((double)(3.0 * Variance) / Radius);
			for (i = 0; i <= Radius; ++i)
			{
				double x = i * xRatio;
				GaussianTable[i] = (1.0 / (sqrt(2.0 * PI) * Variance)) * pow(e, (-(x * x) / (2 * Variance * Variance)));
			}

			int TotalWeight = 0;
			for (i = 0; i <= Radius; ++i)
			{
				GaussianTable_Int[i] = GaussianTable[i] / GaussianTable[Radius];
				TotalWeight += GaussianTable_Int[i];
			}

			Mat Temp(SourceImage.rows, SourceImage.cols, SourceImage.type());
			uchar *pSource, *pTemp;
			for (i = 0; i < SourceImage.rows; i++)
			{
				pTemp = Temp.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(i);
				for (j = 0; j < SourceImage.cols * SourceImage.channels(); ++j)
				{

				}
			}










			SourceImage.copyTo(Result);

			imshow("快速高斯滤波", Result);
		}
	}FastGaussianFilter;

	FastGaussianFilter.Filter(0, 0);

	createTrackbar("半径", "快速高斯滤波", &Radius, (SourceImage.cols > SourceImage.rows ? SourceImage.rows : SourceImage.cols) * 0.25, FastGaussianFilter.Filter);
	createTrackbar("方差", "快速高斯滤波", &Variance, 5, FastGaussianFilter.Filter);

}
