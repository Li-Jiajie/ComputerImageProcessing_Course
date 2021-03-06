
// Computer Image ProcessingDlg.h: 头文件
//

#pragma once


// CComputerImageProcessingDlg 对话框
class CComputerImageProcessingDlg : public CDialogEx
{
// 构造
public:
	CComputerImageProcessingDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_COMPUTERIMAGEPROCESSING_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnOpenImageFile();
	afx_msg void OnAllRed();
	afx_msg void OnRedBlue();
	afx_msg void OnGray();
	afx_msg void OnImageZoom_Normal();
	afx_msg void OnImageZoom_Linear();
	afx_msg void OnImageZoom_Cubic();
	afx_msg void OnBinaryzation_GivenThreshold();
	afx_msg void OnHistogramEqualization();
	afx_msg void OnInvert();
	afx_msg void OnCustomHistogramMapping();
	afx_msg void OnBitPlaneSlicing();
	afx_msg void OnFastGaussianFilter();
	afx_msg void OnFunctionTest();
	afx_msg void OnMedianFilter_FullMatrix();
	afx_msg void OnMedianFilter_Cross();
	afx_msg void OnLaplace();
	afx_msg void OnRobert();
	afx_msg void OnSobel();
	afx_msg void OnRGBToHSI();
	afx_msg void OnGrayToFalseColor();
	afx_msg void OnEroding();
	afx_msg void OnDilating();
	afx_msg void OnImageOpen();
	afx_msg void OnImageClose();
	afx_msg void OnGetBorder_Morphological();
	afx_msg void OnFill_Morphological();
	afx_msg void OnConnectedComponent();
	afx_msg void OnHuffman();
	afx_msg void OnOTSU();
};
