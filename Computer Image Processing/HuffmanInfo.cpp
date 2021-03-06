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
	//DDX_Control(pDX, IDC_EDIT1, infotext);
}


BEGIN_MESSAGE_MAP(HuffmanInfo, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON1, &HuffmanInfo::OnBnClickedButton1)
END_MESSAGE_MAP()


// HuffmanInfo 消息处理程序


void HuffmanInfo::OnBnClickedButton1()
{
	((CListBox*)GetDlgItem(IDC_LIST1))->ResetContent();
	for (int i = 0; i < 256; ++i)
	{
		char content[500];

		sprintf(content, "灰度:%7s   编码:%s", HuffmanContent[i].gray.c_str(), HuffmanContent[i].encrypt.c_str());
		((CListBox*)GetDlgItem(IDC_LIST1))->InsertString(i, content);
	}
	((CEdit*)GetDlgItem(IDC_EDIT1))->SetWindowTextA(std::to_string(TotalLength).c_str());
	((CEdit*)GetDlgItem(IDC_EDIT3))->SetWindowTextA(std::to_string((double)TotalLength / Size).c_str());
	((CEdit*)GetDlgItem(IDC_EDIT4))->SetWindowTextA(std::to_string(Size).c_str());
	((CEdit*)GetDlgItem(IDC_EDIT5))->SetWindowTextA(std::to_string(1.0 - ((double)TotalLength / Size) / 8.0).c_str());

}
