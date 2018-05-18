// HuffmanInfo.cpp: 实现文件
//

#include "stdafx.h"
#include "Computer Image Processing.h"
#include "HuffmanInfo.h"
#include "afxdialogex.h"

// HuffmanInfo 对话框

IMPLEMENT_DYNAMIC(HuffmanInfo, CDialogEx)

HuffmanInfo::HuffmanInfo(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG2, pParent)
{
}

HuffmanInfo::~HuffmanInfo()
{
}

void HuffmanInfo::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT1, infotext);
}


BEGIN_MESSAGE_MAP(HuffmanInfo, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON1, &HuffmanInfo::OnBnClickedButton1)
END_MESSAGE_MAP()


// HuffmanInfo 消息处理程序


void HuffmanInfo::OnBnClickedButton1()
{
	for (int i = 0; i < 256; ++i)
	{
		((CListBox*)GetDlgItem(IDC_LIST1))->AddString(HuffmanContent[i].gray + "- " + HuffmanContent[i].encrypt);
	}
}
