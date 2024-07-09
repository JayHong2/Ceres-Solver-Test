﻿
// AF_Fitting2View.h: CAFFitting2View 클래스의 인터페이스
//

#pragma once


class CAFFitting2View : public CFormView
{
protected: // serialization에서만 만들어집니다.
	CAFFitting2View() noexcept;
	DECLARE_DYNCREATE(CAFFitting2View)

public:
#ifdef AFX_DESIGN_TIME
	enum{ IDD = IDD_AF_FITTING2_FORM };
#endif

// 특성입니다.
public:
	CAFFitting2Doc* GetDocument() const;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.
	virtual void OnInitialUpdate(); // 생성 후 처음 호출되었습니다.

// 구현입니다.
public:
	virtual ~CAFFitting2View();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonTest();
	afx_msg void OnBnClickedButtonTestAlignMaterix();
	afx_msg void OnBnClickedButtonTestInverseAlignMaterix();
};

#ifndef _DEBUG  // AF_Fitting2View.cpp의 디버그 버전
inline CAFFitting2Doc* CAFFitting2View::GetDocument() const
   { return reinterpret_cast<CAFFitting2Doc*>(m_pDocument); }
#endif

