
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
	ON_COMMAND(ID_32772, &CComputerImageProcessingDlg::On32772)
	ON_COMMAND(ID_32773, &CComputerImageProcessingDlg::OnRedBlue)
	ON_COMMAND(ID_32774, &CComputerImageProcessingDlg::OnGray)
	ON_COMMAND(ID_32776, &CComputerImageProcessingDlg::OnImageZoom_Normal)
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


void CComputerImageProcessingDlg::On32772()
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


void CComputerImageProcessingDlg::OnImageZoom_Normal()
{
	//static Mat Result;
	static int RowSlider, ColSlider;
	RowSlider = 50;
	ColSlider = 50;
	//Result = SourceImage;

	namedWindow("Result"); 
	imshow("Result", SourceImage);

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
			int x;

			int* MapCol = new int[Result.cols];
			for (i = 0; i < Result.cols; ++i)
			{
				MapCol[i] = (int)(i / ColScaling) * 3;
			}

			for (i = 0; i < Result.rows; i++)
			{
				x = i / RowScaling;
				pResult = Result.ptr<uchar>(i);
				pSource = SourceImage.ptr<uchar>(x);
				for (j = 0; j < Result.cols; ++j)
				{
					*pResult++ = pSource[MapCol[j]];
					*pResult++ = pSource[MapCol[j] + 1];
					*pResult++ = pSource[MapCol[j] + 2];
				}
			}

			imshow("Result", Result);
		}
	}Zoom_Normal;
	createTrackbar("宽度缩放", "Result", &ColSlider, 100, Zoom_Normal.Zoom);
	createTrackbar("高度缩放", "Result", &RowSlider, 100, Zoom_Normal.Zoom);


}
