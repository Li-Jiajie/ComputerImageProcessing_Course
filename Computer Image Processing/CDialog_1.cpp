// CDialog_1.cpp: 实现文件
//

#include "stdafx.h"
#include "Computer Image Processing.h"
#include "CDialog_1.h"
#include "afxdialogex.h"


// CDialog_1 对话框

IMPLEMENT_DYNAMIC(CDialog_1, CDialogEx)

CDialog_1::CDialog_1(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG1, pParent)
{

}

CDialog_1::~CDialog_1()
{
}

void CDialog_1::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CDialog_1, CDialogEx)
	ON_BN_CLICKED(IDOK, &CDialog_1::OnClickOK)
END_MESSAGE_MAP()


void CDialog_1::OnClickOK()
{
	CString NumberA, NumberB;
	GetDlgItem(IDC_EDIT1)->GetWindowTextA(NumberA);
	GetDlgItem(IDC_EDIT2)->GetWindowTextA(NumberB);
	double A, B;
    A = (float)atof((char *)(LPTSTR)(LPCTSTR)NumberA);
	B = (float)atof((char *)(LPTSTR)(LPCTSTR)NumberB);

	int i, j;

	double Max = 0.00001;
	for (i = 0; i < 256; ++i)
	{
		MappingTable[i] = pow((A * i / 255), B);
		if (fabs(MappingTable[i]) > Max)
			Max = MappingTable[i];
	}
	for (i = 0; i < 256; ++i)
	{
		MappingTable[i] = (MappingTable[i] / Max * 255);
	}

	OnOK();
}
