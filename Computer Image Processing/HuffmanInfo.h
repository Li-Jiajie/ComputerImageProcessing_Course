#pragma once
// HuffmanInfo 对话框
#include <string>
struct Huffman_Content
{
	std::string gray="";
	std::string encrypt="";
};

class HuffmanInfo : public CDialogEx
{
	DECLARE_DYNAMIC(HuffmanInfo)
	
public:
	Huffman_Content HuffmanContent[256];
	int TotalLength;
	int Size;

	HuffmanInfo(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~HuffmanInfo();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DIALOG2 };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	CEdit infotext;
	afx_msg void OnBnClickedButton1();
};
