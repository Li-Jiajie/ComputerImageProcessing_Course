#pragma once


// CDialog_1 对话框

class CDialog_1 : public CDialogEx
{
	DECLARE_DYNAMIC(CDialog_1)

public:
	CDialog_1(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CDialog_1();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG1 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnClickOK();
	double MappingTable[256];

};
