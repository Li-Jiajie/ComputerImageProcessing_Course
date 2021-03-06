
// Computer Image ProcessingDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "Computer Image Processing.h"
#include "Computer Image ProcessingDlg.h"
#include "afxdialogex.h"
#include <iostream>
#include <list>
#include <cv.h>
#include <cv.hpp>
#include <omp.h>
#include <thread>
#include "ProcessFunction.h"
#include "CDialog_1.h"
#include "HuffmanInfo.h"

using namespace std;
using namespace cv;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

HuffmanInfo *m_dlg;


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
	m_dlg = NULL;
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
	ON_COMMAND(ID_32785, &CComputerImageProcessingDlg::OnFunctionTest)
	ON_COMMAND(ID_32786, &CComputerImageProcessingDlg::OnMedianFilter_FullMatrix)
	ON_COMMAND(ID_32787, &CComputerImageProcessingDlg::OnMedianFilter_Cross)
	ON_COMMAND(ID_32788, &CComputerImageProcessingDlg::OnLaplace)
	ON_COMMAND(ID_32789, &CComputerImageProcessingDlg::OnRobert)
	ON_COMMAND(ID_32790, &CComputerImageProcessingDlg::OnSobel)
	ON_COMMAND(ID_32791, &CComputerImageProcessingDlg::OnRGBToHSI)
	ON_COMMAND(ID_32792, &CComputerImageProcessingDlg::OnGrayToFalseColor)
	ON_COMMAND(ID_32794, &CComputerImageProcessingDlg::OnEroding)
	ON_COMMAND(ID_32795, &CComputerImageProcessingDlg::OnDilating)
	ON_COMMAND(ID_32796, &CComputerImageProcessingDlg::OnImageOpen)
	ON_COMMAND(ID_32797, &CComputerImageProcessingDlg::OnImageClose)
	ON_COMMAND(ID_32798, &CComputerImageProcessingDlg::OnGetBorder_Morphological)
	ON_COMMAND(ID_32799, &CComputerImageProcessingDlg::OnFill_Morphological)
	ON_COMMAND(ID_32800, &CComputerImageProcessingDlg::OnConnectedComponent)
	ON_COMMAND(ID_32801, &CComputerImageProcessingDlg::OnHuffman)
	ON_COMMAND(ID_32802, &CComputerImageProcessingDlg::OnOTSU)
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


void CComputerImageProcessingDlg::OnFunctionTest()
{
	Mat Source(Size(4, 5), CV_8U, Scalar(0));
	Source.at<uchar>(1, 1) = 255;
	Source.at<uchar>(1, 2) = 255;
	Source.at<uchar>(2, 1) = 255;
	Source.at<uchar>(2, 2) = 255;
	Source.at<uchar>(3, 1) = 255;
	Source.at<uchar>(3, 2) = 255;

	Mat Result;
	Mat Kernel(Size(2, 2), CV_8U);

	Kernel.at<uchar>(0, 0) = 0;
	Kernel.at<uchar>(0, 1) = 1;
	Kernel.at<uchar>(1, 0) = 1;
	Kernel.at<uchar>(1, 1) = 1;

	dilate(Source, Result, Kernel, Point(1, 1), 1);

	resize(Source, Source, Size(Source.cols * 100, Source.rows * 100), 0, 0, INTER_NEAREST);
	resize(Result, Result, Size(Result.cols * 100, Result.rows * 100), 0, 0, INTER_NEAREST);

	imshow("Source", Source);
	imshow("Result", Result);

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

	if(SourceImage.channels() != 1)
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
			for (i = 0; i < Result.rows - 1; i++)
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
				//2018.6.2注：该处可优化 对行列彻底分开计算可减少运算量 实际上本算法忽略了最后一行也是由当前取下一行值造成的。由于已期末且算法可运行，暂不修改。
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
			resize(SourceImage, Result, Size(ColScaling * SourceImage.cols, RowScaling * SourceImage.rows), 0, 0, INTER_CUBIC);

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
	if (SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, CV_RGB2GRAY);

	imshow("Source", SourceImage);

	Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

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
		PixelRate[i] = (double)PixelCount[i] / PixelNumber + PixelRate[i - 1];
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

	imshow("Source", SourceImage);
	namedWindow("快速高斯滤波");

	static int Radius, Variance;
	Radius = 8;
	Variance = 1;

	struct
	{
		static void Filter(int Pos, void* ext)
		{
			if (Variance == 0)
			{
				AfxMessageBox(_T("正态分布函数方差不能为0！"));
				Variance = 1;
			}
			Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

			double *GaussianTable = new double[Radius + 1];
			int *GaussianTable_Int = new int[Radius + 1];
			int i, j;
			int ImageChannel = SourceImage.channels();

			double xRatio = (double)((double)(3.0 * Variance) / Radius);

			if (Radius == 0)
				xRatio = 0;

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
			//行处理
			for (i = 0; i < SourceImage.rows; i++)
			{
				pTemp = Temp.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(i);
#pragma omp parallel for
				for (j = 0; j < SourceImage.cols; ++j)
				{
					for (int channel = 0; channel < ImageChannel; ++channel)
					{
						int begin = j - Radius, end = j + Radius;
						if (begin < 0)
							begin = 0;
						if (end > SourceImage.cols)
							end = SourceImage.cols;
						int count = 0;
						int ColorSum = 0;
						for (int cols = begin; cols <= end; ++cols)
						{
							int Weight = GaussianTable_Int[abs(cols - j)];
							count += Weight;
							ColorSum += pSource[cols * ImageChannel + channel] * Weight;
						}
						pTemp[j * ImageChannel + channel] = ColorSum / count;
					}
				}
			}

			//列处理
			uchar *ColPixelArray = new uchar[SourceImage.rows * ImageChannel];
			for (i = 0; i < SourceImage.cols; i++)
			{
				//将列像素存到临时数组中以减少未来访问时间
				for (int pixel = 0; pixel < SourceImage.rows; ++pixel)
				{
					memcpy(ColPixelArray + pixel * ImageChannel, Temp.ptr<uchar>(pixel) + i * ImageChannel, sizeof(uchar) * ImageChannel);
				}
#pragma omp parallel for
				for (j = 0; j < SourceImage.rows; ++j)
				{
					for (int channel = 0; channel < ImageChannel; ++channel)
					{

						int begin = j - Radius, end = j + Radius;
						if (begin < 0)
							begin = 0;
						if (end > SourceImage.rows)
							end = SourceImage.rows;
						int count = 0;
						int ColorSum = 0;
						for (int rows = begin; rows <= end; ++rows)
						{
							int Weight = GaussianTable_Int[abs(rows - j)];
							count += Weight;
							ColorSum += ColPixelArray[rows * ImageChannel + channel] * Weight;
						}
						Result.ptr<uchar>(j)[i * ImageChannel + channel] = ColorSum / count;
					}
				}
			}

			delete[] ColPixelArray;
			delete[] GaussianTable;
			delete[] GaussianTable_Int;

			imshow("快速高斯滤波", Result);
		}
	}FastGaussianFilter;

	FastGaussianFilter.Filter(0, 0);

	createTrackbar("半径", "快速高斯滤波", &Radius, (SourceImage.cols > SourceImage.rows ? SourceImage.rows : SourceImage.cols) * 0.25, FastGaussianFilter.Filter);
	createTrackbar("方差", "快速高斯滤波", &Variance, 5, FastGaussianFilter.Filter);
}



void CComputerImageProcessingDlg::OnMedianFilter_FullMatrix()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	imshow("Source", SourceImage);
	namedWindow("2D中值滤波(全矩阵)");


	static int Radius;
	Radius = 1;

	struct
	{
		static int Compare(const void *a, const void *b)
		{
			return *(int*)a - *(int*)b;
		}
		static void FullMatrix(int Pos, void* ext)
		{
			Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

			uchar **pSource, **pResult;
			pSource = new uchar*[SourceImage.rows];
			pResult = new uchar*[Result.rows];

			for (int i = 0; i < SourceImage.rows; ++i)
			{
				pResult[i] = Result.ptr<uchar>(i);
				pSource[i] = SourceImage.ptr<uchar>(i);
			}

			int MatrixSize = (Radius + Radius + 1) * (Radius + Radius + 1);
			int ImageChannel = SourceImage.channels();

			for (int i = 0; i < Result.rows; ++i)
			{
#pragma omp parallel for
				for (int j = 0; j < Result.cols; ++j)
				{
					int *FilterArray = new int[MatrixSize];
					for (int c = 0; c < ImageChannel; ++c)
					{
						int Count = 0;
						int RowStart, RowEnd, ColStart, ColEnd;
						RowStart = i - Radius < 0 ? 0 : i - Radius;
						RowEnd = i + Radius >= Result.rows ? Result.rows - 1 : i + Radius;
						ColStart = j - Radius < 0 ? 0 : j - Radius;
						ColEnd = j + Radius >= Result.cols ? Result.cols - 1 : j + Radius;
						for (int m = RowStart; m <= RowEnd; ++m)
						{
							for (int n = ColStart; n <= ColEnd; ++n)
							{
								FilterArray[Count++] = pSource[m][n * 3 + c];
							}
						}
						//sort(FilterArray, FilterArray + Count);
						qsort(FilterArray, Count, sizeof(int), Compare);
						pResult[i][j * 3 + c] = FilterArray[Count >> 1];
					}
					delete[] FilterArray;

				}
			}

			imshow("2D中值滤波(全矩阵)", Result);
		}
	}MedianFilter;

	MedianFilter.FullMatrix(1, 0);

	createTrackbar("矩阵大小(半径)", "2D中值滤波(全矩阵)", &Radius, 5, MedianFilter.FullMatrix);
}

void CComputerImageProcessingDlg::OnMedianFilter_Cross()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	imshow("Source", SourceImage);
	namedWindow("2D中值滤波(十字稀疏矩阵)");


	static int Radius;
	Radius = 1;

	struct
	{
		static int Compare(const void *a, const void *b)
		{
			return *(int*)a - *(int*)b;
		}
		static void FullMatrix(int Pos, void* ext)
		{
			Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

			uchar **pSource, **pResult;
			pSource = new uchar*[SourceImage.rows];
			pResult = new uchar*[Result.rows];

			for (int i = 0; i < SourceImage.rows; ++i)
			{
				pResult[i] = Result.ptr<uchar>(i);
				pSource[i] = SourceImage.ptr<uchar>(i);
			}

			int MatrixSize = (Radius + Radius + 1) * (Radius + Radius + 1);
			int ImageChannel = SourceImage.channels();

			for (int i = 0; i < Result.rows; ++i)
			{
#pragma omp parallel for
				for (int j = 0; j < Result.cols; ++j)
				{
					int *FilterArray = new int[MatrixSize];
					for (int c = 0; c < ImageChannel; ++c)
					{
						int Count = 0;
						int RowStart, RowEnd, ColStart, ColEnd;
						RowStart = i - Radius < 0 ? 0 : i - Radius;
						RowEnd = i + Radius >= Result.rows ? Result.rows - 1 : i + Radius;
						ColStart = j - Radius < 0 ? 0 : j - Radius;
						ColEnd = j + Radius >= Result.cols ? Result.cols - 1 : j + Radius;
						for (int m = RowStart; m <= RowEnd; ++m)
						{
							if (m == i)
								continue;
							FilterArray[Count++] = pSource[m][j * 3 + c];
						}
						for (int m = ColStart; m <= ColEnd; ++m)
						{
							FilterArray[Count++] = pSource[i][m * 3 + c];
						}
						//sort(FilterArray, FilterArray + Count);
						qsort(FilterArray, Count, sizeof(int), Compare);
						pResult[i][j * 3 + c] = FilterArray[Count >> 1];
					}
					delete[] FilterArray;

				}
			}

			imshow("2D中值滤波(十字稀疏矩阵)", Result);
		}
	}MedianFilter;

	MedianFilter.FullMatrix(1, 0);

	createTrackbar("矩阵大小(半径)", "2D中值滤波(十字稀疏矩阵)", &Radius, 5, MedianFilter.FullMatrix);
}


void CComputerImageProcessingDlg::OnLaplace()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	Mat Result;
	SourceImage.copyTo(Result);
	int ImageChannel = Result.channels();

	for (int i = 1; i < Result.rows - 1; ++i)
	{
		uchar *Up = SourceImage.ptr<uchar>(i - 1);
		uchar *Current = SourceImage.ptr<uchar>(i);
		uchar *Down = SourceImage.ptr<uchar>(i + 1);
		uchar *pResult = Result.ptr<uchar>(i);

		for (int j = 1; j < Result.cols - 1; ++j)
		{
			for (int c = 0; c < ImageChannel; ++c)
			{
				int Pos = j * ImageChannel + c;
				*pResult++ = saturate_cast<uchar>(5 * Current[Pos] - Current[Pos - ImageChannel] - Current[Pos + ImageChannel] - Up[Pos] - Down[Pos]);
			}
		}
	}

	imshow("锐化(拉普拉斯)", Result);
}


void CComputerImageProcessingDlg::OnRobert()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	if (SourceImage.channels() != 1)
	{
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
		imshow("Source", SourceImage);
	}
	Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

	for (int i = 0; i < Result.rows - 1; ++i)
	{
		uchar *Current = SourceImage.ptr<uchar>(i);
		uchar *Down = SourceImage.ptr<uchar>(i + 1);
		uchar *pResult = Result.ptr<uchar>(i);

		for (int j = 1; j < Result.cols - 1; ++j)
		{
			int RobertA, RobertB;
			RobertA = pow((Current[j] - Down[j + 1]), 2);
			RobertB = pow((Current[j + 1] - Down[j]), 2);
			*pResult++ = sqrt(RobertA + RobertB);
		}
	}

	imshow("锐化(Robert)", Result);
}


void CComputerImageProcessingDlg::OnSobel()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	if (SourceImage.channels() != 1)
	{
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
		imshow("Source", SourceImage);
	}

	Mat SobelX(SourceImage.rows, SourceImage.cols, SourceImage.type());
	Mat SobelY(SourceImage.rows, SourceImage.cols, SourceImage.type());
	Mat SobelXY(SourceImage.rows, SourceImage.cols, SourceImage.type());

	for (int i = 1; i < SourceImage.rows - 1; ++i)
	{
		uchar *Up = SourceImage.ptr<uchar>(i - 1);
		uchar *Current = SourceImage.ptr<uchar>(i);
		uchar *Down = SourceImage.ptr<uchar>(i + 1);
		uchar *pX = SobelX.ptr<uchar>(i);
		uchar *pY = SobelY.ptr<uchar>(i);

		for (int j = 1; j < SourceImage.cols - 1; ++j)
		{
			*pX++ = saturate_cast<uchar>(abs(-Up[j - 1] + Up[j + 1] - 2 * Current[j - 1] + 2 * Current[j + 1] - Down[j - 1] + Down[j + 1]));
			*pY++ = saturate_cast<uchar>(abs(Up[j - 1] + 2 * Up[j] + Up[j + 1] - Down[j - 1] - 2 * Down[j] - Down[j + 1]));
		}
	}

	addWeighted(SobelX, 0.5, SobelY, 0.5, 0, SobelXY);
	imshow("锐化(Sobel XY)", SobelXY);
	imshow("锐化(Sobel X)", SobelX);
	imshow("锐化(Sobel Y)", SobelY);
}


void CComputerImageProcessingDlg::OnRGBToHSI()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	if (SourceImage.channels() == 1)
	{
		AfxMessageBox(_T("请使用三通道彩色图像！"));
		return;
	}

	double Normalization[256];
	for (int i = 0; i < 256; ++i)
	{
		Normalization[i] = i / 255.0;
	}

	struct HSI
	{
		double H, S, I;
	};

	static HSI **Image_HSI = new HSI*[SourceImage.rows];
	for (int i = 0; i < SourceImage.rows; ++i)
		Image_HSI[i] = new HSI[SourceImage.cols];

#pragma omp parallel for
	for (int i = 0; i < SourceImage.rows; ++i)
	{
		uchar *pSource = SourceImage.ptr<uchar>(i);
		double r, g, b, min, angle;
		for (int j = 0; j < SourceImage.cols; ++j)
		{
			b = Normalization[*pSource++];
			g = Normalization[*pSource++];
			r = Normalization[*pSource++];
			min = b < g ? b : g;
			min = min < r ? min : r;
			Image_HSI[i][j].I = (r + g + b) / 3.0;
			Image_HSI[i][j].S = 1.0 - (3.0 * min) / (r + b + g);
			angle = acos((r - g + r - b) / (2 * sqrt((r - g) * (r - g) + (r - b) * (g - b))));
			if (g >= b)
				Image_HSI[i][j].H = angle;
			else
				Image_HSI[i][j].H = 2 * PI - angle;
		}
	}

	Mat H(SourceImage.rows, SourceImage.cols, CV_8U);
	Mat S(SourceImage.rows, SourceImage.cols, CV_8U);
	Mat I(SourceImage.rows, SourceImage.cols, CV_8U);

	for (int i = 0; i < SourceImage.rows; ++i)
	{
		uchar *pH = H.ptr<uchar>(i);
		uchar *pS = S.ptr<uchar>(i);
		uchar *pI = I.ptr<uchar>(i);
		for (int j = 0; j < SourceImage.cols; ++j)
		{
			*pH++ = Image_HSI[i][j].H * 255.0;
			*pS++ = Image_HSI[i][j].S * 255.0;
			*pI++ = Image_HSI[i][j].I * 255.0;
		}
	}

	imshow("h", H);
	imshow("s", S);
	imshow("I", I);

	namedWindow("HSI");
	imshow("HSI", SourceImage);

	static int H_Bar = 0, S_Bar = 50, I_Bar = 50;
	static double R60 = (PI * 2.0 * (60.0 / 360.0)), R120 = (PI * 2.0 * (120.0 / 360.0)), R240 = (PI * 2.0 * (240.0 / 360.0));
	struct
	{
		static void Process(int Pos, void* ext)
		{
			Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());

			double Ratio_S, Ratio_I;
			Ratio_S = (double)(S_Bar - 50) / 50.0 + 1;
			Ratio_I = (double)(I_Bar - 50) / 50.0 + 1;

#pragma omp parallel for
			for (int i = 0; i < SourceImage.rows; ++i)
			{
				uchar *pResult = Result.ptr<uchar>(i);
				for (int j = 0; j < SourceImage.cols; ++j)
				{
					HSI NewHSI;
					NewHSI.H = Image_HSI[i][j].H + (H_Bar / 180.0 * PI);
					if (NewHSI.H > 2.0 * PI)
						NewHSI.H -= 2.0 * PI;
					else if (NewHSI.H < 0)
						NewHSI.H += 2.0 * PI;

					NewHSI.I = Image_HSI[i][j].I * Ratio_I;
					NewHSI.S = Image_HSI[i][j].S * Ratio_S;

					double r = 0, g = 0, b = 0;
					if (NewHSI.H > 0 && NewHSI.H <= R120)
					{
						r = NewHSI.I * (1.0 + (NewHSI.S * cos(NewHSI.H)) / (cos(R60 - NewHSI.H)));
						b = NewHSI.I * (1.0 - NewHSI.S);
						g = 3 * NewHSI.I - r - b;
					}
					else if (NewHSI.H >= R120 && NewHSI.H <= R240)
					{
						NewHSI.H -= R120;
						g = NewHSI.I * (1 + (NewHSI.S * cos(NewHSI.H)) / (cos(R60 - NewHSI.H)));
						r = NewHSI.I * (1 - NewHSI.S);
						b = 3 * NewHSI.I - r - g;
					}
					else
					{
						NewHSI.H -= R240;
						b = NewHSI.I * (1 + (NewHSI.S * cos(NewHSI.H)) / (cos(R60 - NewHSI.H)));
						g = NewHSI.I * (1 - NewHSI.S);
						r = 3 * NewHSI.I - g - b;
					}

					*pResult++ = saturate_cast<uchar>(b * 255);
					*pResult++ = saturate_cast<uchar>(g * 255);
					*pResult++ = saturate_cast<uchar>(r * 255);
				}
			}

			imshow("HSI", Result);
		}
	}HSIProcess;

	createTrackbar("H分量角度", "HSI", &H_Bar, 360, HSIProcess.Process);
	createTrackbar("S分量", "HSI", &S_Bar, 100, HSIProcess.Process);
	createTrackbar("I分量", "HSI", &I_Bar, 100, HSIProcess.Process);
}

void CComputerImageProcessingDlg::OnGrayToFalseColor()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	if (SourceImage.channels() != 1)
	{
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
		imshow("Source", SourceImage);
	}

	Mat Result(SourceImage.rows, SourceImage.cols, CV_8UC3);

	uchar ColorMap[256][3];

	for (int i = 0; i < 64; ++i)
	{
		ColorMap[i][0] = 255;
		ColorMap[i][1] = (i + 1) / 64.0 * 255.0;
		ColorMap[i][2] = 0;
	}
	for (int i = 64; i < 128; ++i)
	{
		ColorMap[i][0] = (129 - i) / 64.0 * 255.0;
		ColorMap[i][1] = 255;
		ColorMap[i][2] = 0;
	}
	for (int i = 128; i < 192; ++i)
	{
		ColorMap[i][0] = 0;
		ColorMap[i][1] = (193 - i) / 64.0 * 255.0;
		ColorMap[i][2] = (i - 127) / 64.0 * 255.0;
	}
	for (int i = 192; i < 256; ++i)
	{
		ColorMap[i][0] = (i - 191) / 64.0 * 255.0;
		ColorMap[i][1] = 0;
		ColorMap[i][2] = 255;
	}

	for (int i = 0; i < SourceImage.rows; ++i)
	{
		uchar *pSource = SourceImage.ptr<uchar>(i);
		uchar *pResult = Result.ptr<uchar>(i);
		for (int j = 0; j < SourceImage.cols; ++j)
		{
			*pResult++ = ColorMap[*pSource][2];
			*pResult++ = ColorMap[*pSource][1];
			*pResult++ = ColorMap[*pSource++][0];
		}
	}

	imshow("灰度转伪彩色图像", Result);
}

void CComputerImageProcessingDlg::OnEroding()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	namedWindow("腐蚀");
	static int Radius;
	Radius = 3;
	struct
	{
		static void Function(int Pos, void* ext)
		{
			if (Radius == 0)
				Radius = 1;

			Mat Result;
			Mat Kernel(Size(Radius, Radius), CV_8U, Scalar(1));
			Erosion(SourceImage, Result, Kernel, Point(Radius * 0.5, Radius * 0.5));
			imshow("腐蚀", Result);
		}
	}Eroding;

	Eroding.Function(1, 0);
	createTrackbar("内核半径", "腐蚀", &Radius, 20, Eroding.Function);
}


void CComputerImageProcessingDlg::OnDilating()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	namedWindow("膨胀");
	static int Radius;
	Radius = 3;
	struct
	{
		static void Function(int Pos, void* ext)
		{
			if (Radius == 0)
				Radius = 1;

			Mat Result;
			Mat Kernel(Size(Radius, Radius), CV_8U, Scalar(1));
			Dilation(SourceImage, Result, Kernel, Point(Radius * 0.5, Radius * 0.5));
			imshow("膨胀", Result);
		}
	}Dilating;

	Dilating.Function(1, 0);
	createTrackbar("内核半径", "膨胀", &Radius, 20, Dilating.Function);
}


void CComputerImageProcessingDlg::OnImageOpen()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	namedWindow("开操作");

	static int Ero, Dia;
	Ero = 3;
	Dia = 3;

	struct
	{
		static void Function(int Pos, void* ext)
		{
			if (Ero == 0)
				Ero = 1;
			if (Dia == 0)
				Dia = 1;

			Mat Result;

			Mat ErosionKernel(Size(Ero, Ero), CV_8U, Scalar(1));
			Mat DilationKernel(Size(Dia, Dia), CV_8U, Scalar(1));

			Erosion(SourceImage, Result, ErosionKernel, Point(Ero * 0.5, Ero * 0.5));
			Dilation(Result, Result, DilationKernel, Point(Dia * 0.5, Dia * 0.5));
			imshow("开操作", Result);
		}
	}Opening;

	Opening.Function(1, 0);

	createTrackbar("腐蚀半径", "开操作", &Ero, 20, Opening.Function);
	createTrackbar("膨胀半径", "开操作", &Dia, 20, Opening.Function);
}


void CComputerImageProcessingDlg::OnImageClose()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	namedWindow("闭操作");

	static int Ero, Dia;
	Ero = 3;
	Dia = 3;

	struct
	{
		static void Function(int Pos, void* ext)
		{
			if (Ero == 0)
				Ero = 1;
			if (Dia == 0)
				Dia = 1;

			Mat Result;

			Mat ErosionKernel(Size(Ero, Ero), CV_8U, Scalar(1));
			Mat DilationKernel(Size(Dia, Dia), CV_8U, Scalar(1));

			Dilation(SourceImage, Result, ErosionKernel, Point(Ero * 0.5, Ero * 0.5));
			Erosion(Result, Result, DilationKernel, Point(Dia * 0.5, Dia * 0.5));
			imshow("闭操作", Result);
		}
	}Closing;

	Closing.Function(1, 0);

	createTrackbar("腐蚀半径", "闭操作", &Ero, 20, Closing.Function);
	createTrackbar("膨胀半径", "闭操作", &Dia, 20, Closing.Function);
}


void CComputerImageProcessingDlg::OnGetBorder_Morphological()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}

	Mat Threshold;
	cvtColor(SourceImage, Threshold, COLOR_BGR2GRAY);
	threshold(Threshold, Threshold, 0, 255, THRESH_OTSU);

	Mat ErosionImage;
	Mat ErosionKernel(Size(3, 3), CV_8U, Scalar(1));
	Erosion(Threshold, ErosionImage, ErosionKernel, Point(1, 1));

	Threshold = Threshold - ErosionImage;

	imshow("边界提取", Threshold);
}


void CComputerImageProcessingDlg::OnFill_Morphological()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
	threshold(SourceImage, SourceImage, 0, 255, THRESH_OTSU);
	imshow("区域填充", SourceImage);

	static Mat Result;
	SourceImage.copyTo(Result);

	struct
	{
		static void Function(int event, int x, int y, int flag, void* param)
		{
			switch (event)
			{
			case EVENT_LBUTTONUP:
			{
				Mat Kernel(Size(3, 3), CV_8U, Scalar(1));
				Kernel.at<uchar>(0, 0) = 0;
				Kernel.at<uchar>(2, 2) = 0;
				Kernel.at<uchar>(0, 2) = 0;
				Kernel.at<uchar>(2, 0) = 0;

				Mat dst = Mat::zeros(Result.size(), Result.type());
				Mat tempImg = Mat::ones(Result.size(), Result.type()) * 255;
				Mat revImg = tempImg - Result;//原图像的补集  
				dst.at<uchar>(y, x) = 255;//绘制种子点  

				while (true)
				{
					Mat Temp;
					dst.copyTo(Temp);
					Dilation(dst, dst, Kernel, Point(1, 1));
					dst = dst & revImg;

					if (memcmp(Temp.data, dst.data, dst.total() * dst.elemSize()) == 0)
					{
						Result = Result | dst;
						break;
					}
				}
				imshow("区域填充", Result);
				break;
			}
			default:
				break;
			}
		}
	}Fill;

	setMouseCallback("区域填充", Fill.Function);
}


void CComputerImageProcessingDlg::OnConnectedComponent()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
	threshold(SourceImage, SourceImage, 0, 255, THRESH_OTSU);
	imshow("连通分量", SourceImage);

	static Mat Result;
	SourceImage.copyTo(Result);

	struct
	{
		static void Function(int event, int x, int y, int flag, void* param)
		{
			switch (event)
			{
			case EVENT_LBUTTONUP:
			{
				if (Result.at<uchar>(y, x) == 255)
				{
					Result = 255 - Result;
				}
				Mat Kernel(Size(3, 3), CV_8U, Scalar(1));
				Kernel.at<uchar>(0, 0) = 0;
				Kernel.at<uchar>(2, 2) = 0;
				Kernel.at<uchar>(0, 2) = 0;
				Kernel.at<uchar>(2, 0) = 0;

				Mat dst = Mat::zeros(Result.size(), Result.type());
				Mat tempImg = Mat::ones(Result.size(), Result.type()) * 255;
				Mat revImg = tempImg - Result;//原图像的补集  
				dst.at<uchar>(y, x) = 255;//绘制种子点  

				while (true)
				{
					Mat Temp;
					dst.copyTo(Temp);
					Dilation(dst, dst, Kernel, Point(1, 1));
					dst = dst & revImg;

					if (memcmp(Temp.data, dst.data, dst.total() * dst.elemSize()) == 0)
					{
						break;
					}
				}
				imshow("连通分量", dst);
				break;
			}
			default:
				break;
			}
		}
	}ConnectedComponent;

	setMouseCallback("连通分量", ConnectedComponent.Function);
}


void CComputerImageProcessingDlg::OnHuffman()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
	{
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
		imshow("Source", SourceImage);
	}

	struct HuffmanTreeNode;
	struct Encode
	{
		string Code;
		int pixel;
		int count;
		HuffmanTreeNode *Node;
	};
	static vector<Encode*> PictureEncode;

	struct HuffmanTreeNode
	{
		static bool Compare(HuffmanTreeNode *&a, HuffmanTreeNode *&b)
		{
			return a->weight > b->weight;
		}
		static void Traversal(HuffmanTreeNode *Node, string Encrypt)
		{
			if (Node->left != nullptr)
			{
				HuffmanTreeNode::Traversal(Node->left, Encrypt + "0");
			}
			if (Node->right != nullptr)
			{
				HuffmanTreeNode::Traversal(Node->right, Encrypt + "1");
			}
			if (Node->left == nullptr && Node->right == nullptr)
			{
				Encode *NodeCode = new Encode;
				NodeCode->Code = Encrypt;
				NodeCode->count = Node->weight;
				NodeCode->pixel = Node->pixel;
				NodeCode->Node = Node;
				PictureEncode.push_back(NodeCode);
			}
		}
		HuffmanTreeNode *left;
		HuffmanTreeNode *right;
		unsigned int weight = 0;
		int pixel;
	};

	HuffmanTreeNode *PixelCount[256];

	for (int i = 0; i < 256; ++i)
	{
		HuffmanTreeNode *Node = new HuffmanTreeNode;
		Node->left = nullptr;
		Node->right = nullptr;
		Node->weight = 0;
		Node->pixel = i;
		PixelCount[i] = Node;
	}

	for (int i = 0; i < SourceImage.rows; ++i)
	{
		uchar *pSource = SourceImage.ptr<uchar>(i);
		for (int j = 0; j < SourceImage.cols; ++j)
		{
			++PixelCount[*pSource++]->weight;
		}
	}

	int Remain = 256;
	HuffmanTreeNode *HuffmanRoot;
	for (int i = 0; i < 255; ++i)
	{
		sort(PixelCount, PixelCount + Remain, HuffmanTreeNode::Compare);
		HuffmanTreeNode *NewNode = new HuffmanTreeNode;
		NewNode->left = PixelCount[Remain - 1];
		NewNode->right = PixelCount[Remain - 2];
		NewNode->pixel = -1;
		NewNode->weight = NewNode->left->weight + NewNode->right->weight;
		PixelCount[Remain - 2] = NewNode;
		--Remain;
	}
	HuffmanRoot = PixelCount[0];
	PictureEncode.clear();
	HuffmanTreeNode::Traversal(HuffmanRoot, "");

	sort(PictureEncode.begin(), PictureEncode.end(), []
	(Encode *&a, Encode *&b)
	{
		return a->count > b->count;
	});

	if (m_dlg == NULL)
	{
		m_dlg = new HuffmanInfo();
		m_dlg->Create(IDD_DIALOG2, this);
	}

	int TotalLength = 0;
	for (int i = 0; i < PictureEncode.size(); ++i)
	{
		TotalLength += PictureEncode[i]->Code.length() * PictureEncode[i]->count;
		m_dlg->HuffmanContent[i].encrypt = PictureEncode[i]->Code;
		m_dlg->HuffmanContent[i].gray = to_string(PictureEncode[i]->pixel);
	}
	m_dlg->TotalLength = TotalLength;
	m_dlg->Size = SourceImage.cols * SourceImage.rows;

	m_dlg->ShowWindow(SW_SHOW);

	UpdateData(true);

}


void CComputerImageProcessingDlg::OnOTSU()
{
	if (SourceImage.empty())
	{
		AfxMessageBox(_T("请选择源图片！"));
		return;
	}
	if (SourceImage.channels() != 1)
	{
		cvtColor(SourceImage, SourceImage, COLOR_BGR2GRAY);
		imshow("Source", SourceImage);
	}

	int Histogram[256] = {0};
	for (int i = 0; i < SourceImage.rows; ++i)
	{
		uchar *pSource = SourceImage.ptr<uchar>(i);
		for (int j = 0; j < SourceImage.cols; ++j)
		{
			++Histogram[*pSource++];
		}
	}

	double Threshold;
	int ImageLegalSize = SourceImage.rows * SourceImage.cols;
	double Value = 0;
	for (int i = 0; i < 256; ++i)
	{
		double BackgroundProportion = 0;	//背景像素所占比例
		double ForegroundProportion = 0;	//前景像素所占比例
		double BackgroundAve = 0;			//背景平均灰度
		double ForegroundAve = 0;			//前景平均灰度
		for (int j = 0; j <= i; ++j)
		{
			BackgroundProportion += Histogram[j];
			BackgroundAve += j * Histogram[j];
		}
		if (BackgroundProportion == 0)
			continue;
		BackgroundAve /= BackgroundProportion;
		BackgroundProportion /= ImageLegalSize;

		for (int j = i + 1; j < 255; ++j)
		{
			ForegroundProportion += Histogram[j];
			ForegroundAve += j * Histogram[j];
		}
		if (ForegroundProportion == 0)
			continue;
		ForegroundAve /= ForegroundProportion;
		ForegroundProportion /= ImageLegalSize;

		double CurrentValue;
		CurrentValue = ForegroundProportion * BackgroundProportion * (ForegroundAve - BackgroundAve) * (ForegroundAve - BackgroundAve);
		if (CurrentValue > Value)
		{
			Value = CurrentValue;
			Threshold = i;
		}
	}
	
	Mat Result(SourceImage.rows, SourceImage.cols, SourceImage.type());
	for (int i = 0; i < SourceImage.rows; ++i)
	{
		uchar *pResult = Result.ptr<uchar>(i);
		uchar *pSource = SourceImage.ptr<uchar>(i);
		for (int j = 0; j < SourceImage.cols; ++j)
		{
			if (*pSource < Threshold)
				*pResult = 0;
			else
				*pResult = 255;
			++pResult;
			++pSource;
		}
	}

	imshow("OTSU", Result);
}
