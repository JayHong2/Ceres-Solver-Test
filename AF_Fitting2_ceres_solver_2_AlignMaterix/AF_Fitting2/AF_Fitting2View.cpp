
// AF_Fitting2View.cpp: CAFFitting2View 클래스의 구현
//



#include "pch.h"

#define GLOG_USE_GLOG_EXPORT
#define GLOG_NO_ABBREVIATED_SEVERITIES
//


#include ".\\include\glog\\logging.h"
#include ".\\include\ceres\\ceres.h"
#include ".\\include\eigen3\\Eigen\\Core"


#include "AF_Fitting2.h"
#include "AF_Fitting2Doc.h"
#include "AF_Fitting2View.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif





// CAFFitting2View

IMPLEMENT_DYNCREATE(CAFFitting2View, CFormView)

BEGIN_MESSAGE_MAP(CAFFitting2View, CFormView)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_BN_CLICKED(IDC_BUTTON_TEST, &CAFFitting2View::OnBnClickedButtonTest)
	ON_BN_CLICKED(IDC_BUTTON_TEST_ALIGN_MATERIX, &CAFFitting2View::OnBnClickedButtonTestAlignMaterix)
	ON_BN_CLICKED(IDC_BUTTON_TEST_INVERSE_ALIGN_MATERIX, &CAFFitting2View::OnBnClickedButtonTestInverseAlignMaterix)
END_MESSAGE_MAP()

// CAFFitting2View 생성/소멸

CAFFitting2View::CAFFitting2View() noexcept
	: CFormView(IDD_AF_FITTING2_FORM)
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CAFFitting2View::~CAFFitting2View()
{
}

void CAFFitting2View::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
}

BOOL CAFFitting2View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CFormView::PreCreateWindow(cs);
}

void CAFFitting2View::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit();

	// Google 로깅 라이브러리 초기화
	google::InitGoogleLogging("glog started");

	// 로그를 표준 오류(stderr)로 출력하고 파일로 기록하지 않음
	FLAGS_logtostderr = 1;

	// 로그 레벨 설정
	// INFO < WARNING < ERROR < FATAL 순으로 심각도가 증가
	FLAGS_minloglevel = 0; // 모든 로그 레벨 허용

	// default 설정 시, INFO, WARNING 레벨은 로그 파일에만 출력됨
	LOG(INFO) << "INFO 레벨의 로그";
	LOG(WARNING) << "WARNING 레벨의 로그";

	// default 설정 시, ERROR 레벨 이상부터 stderr 로 출력된다.
	LOG(ERROR) << "ERROR 레벨의 로그";

	// FATAL의 경우, Stack trace를 출력하고 프로그램을 종료시킨다.
	//LOG(FATAL) << "FATAL 레벨의 로그";



}

void CAFFitting2View::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CAFFitting2View::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CAFFitting2View 진단

#ifdef _DEBUG
void CAFFitting2View::AssertValid() const
{
	CFormView::AssertValid();
}

void CAFFitting2View::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CAFFitting2Doc* CAFFitting2View::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CAFFitting2Doc)));
	return (CAFFitting2Doc*)m_pDocument;
}
#endif //_DEBUG


// CAFFitting2View 메시지 처리기

using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// 가우시안 모델을 위한 비용 함수
struct GaussianResidual {
	GaussianResidual(double x, double y) : x_(x), y_(y) {}

	template <typename T>
	bool operator()(const T* const offset, const T* const height, const T* const width, const T* const center, T* residual) const {
		// exp(-(x - center)^2 / (2 * width^2))
		// 여기서 T(2)로 수정해야 함
		residual[0] = T(y_) - (offset[0] + height[0] * exp(-(x_ - center[0]) * (x_ - center[0]) / (T(2) * width[0] * width[0])));
		return true;
	}

private:
	const double x_, y_;
};

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
using namespace std;


// 파일 이름을 현재 날짜와 시간으로 생성하는 함수
std::string create_filename() {
	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t(now);
	std::tm buf;
	localtime_s(&buf, &in_time_t);
	std::ostringstream oss;
	oss << std::put_time(&buf, "%Y-%m-%d_%H-%M-%S");
	return oss.str() + "_fit_results.csv";
}

void CAFFitting2View::OnBnClickedButtonTest()
{	
// 	std::vector<double> data = {
// 		0.000416, 0.000417, 0.000416, 0.000416, 0.000416, 0.000417, 0.000416, 0.000416,
// 		0.000416, 0.000418, 0.000418, 0.000417, 0.000417, 0.000420, 0.000417, 0.000416,
// 		0.000418, 0.000416, 0.000417, 0.000419, 0.000416, 0.000417, 0.000418, 0.000417,
// 		0.000417, 0.000419, 0.000417, 0.000418, 0.000418, 0.000417, 0.000418, 0.000418,
// 		0.000418, 0.000419, 0.000419, 0.000418, 0.000419, 0.000419, 0.000419, 0.000419,
// 		0.000420, 0.000420, 0.000421, 0.000421, 0.000421, 0.000425, 0.000423, 0.000424,
// 		0.000425, 0.000424, 0.000424, 0.000427, 0.000426, 0.000428, 0.000429, 0.000429,
// 		0.000431, 0.000433, 0.000432, 0.000434, 0.000435, 0.000435, 0.000437, 0.000438,
// 		0.000439, 0.000442, 0.000444, 0.000446, 0.000449, 0.000452, 0.000455, 0.000459,
// 		0.000464, 0.000470, 0.000477, 0.000482, 0.000489, 0.000500, 0.000511, 0.000528,
// 		0.000558, 0.000594, 0.000620, 0.000693, 0.000754, 0.000809, 0.000829, 0.000815,
// 		0.000782, 0.000732, 0.000682, 0.000640, 0.000605, 0.000574, 0.000549, 0.000529,
// 		0.000510, 0.000497, 0.000487, 0.000480
// 	};

	std::vector<double> data = {
	   4.785869, 4.911828, 4.827335, 4.826616, 4.903931, 4.814974, 4.757457, 4.899952,
	   4.86313, 4.802248, 4.789169, 4.771227, 4.793155, 4.7445, 4.902143, 4.744087,
	   4.685496, 4.763314, 4.591836, 4.868867, 4.689016, 4.638847, 4.698043, 4.764865,
	   4.612763, 4.662041, 4.597139, 4.681048, 4.647728, 4.571094, 4.634619, 4.708748,
	   4.655078, 4.648282, 4.590504, 4.57039, 4.491802, 4.599266, 4.494315, 4.557902,
	   4.548122, 4.424305, 4.546458, 4.524831, 4.476334, 4.43501, 4.47363, 4.500856,
	   4.483276, 4.509808, 4.423032, 4.467703, 4.452664, 4.423568, 4.431982, 4.447869,
	   4.418382, 4.384534, 4.395701, 4.407316, 4.411105, 4.419093, 4.429237, 4.363598,
	   4.35417, 4.379457, 4.34751, 4.404178, 4.353071, 4.326198, 4.347602, 4.354818,
	   4.324647, 4.353156, 4.365736, 4.388994, 4.440554, 4.457387, 4.574845, 4.624974,
	   4.821072, 5.017584, 5.337483, 5.750576, 6.254724, 7.084613, 7.965237, 9.155851,
	   10.889562, 13.18217, 15.76637, 18.510302, 21.105733, 22.221322, 21.349673, 18.927067,
	   16.871291, 13.743958, 12.099179, 9.474953, 8.190292, 7.067767, 6.110413, 5.597851,
	   5.076474, 4.908955, 4.705715, 4.576218, 4.45777, 4.448062, 4.380413, 4.3667,
	   4.378914, 4.370102, 4.351253, 4.31902, 4.344351, 4.342205, 4.321624, 4.346345,
	   4.36062, 4.318589, 4.379205, 4.348644, 4.3821, 4.36239, 4.40071, 4.413266,
	   4.427397, 4.411839, 4.432066, 4.400485, 4.515143, 4.491726, 4.438711, 4.435387,
	   4.45966, 4.414832, 4.47192, 4.501235, 4.478202, 4.512462, 4.418315, 4.488793,
	   4.477456, 4.5371, 4.487586, 4.535939, 4.506829, 4.509452, 4.530762, 4.547781,
	   4.553241, 4.542059, 4.603414, 4.495552, 4.610526, 4.563442, 4.584743, 4.538658,
	   4.637347, 4.697658, 4.59391, 4.562984, 4.679199, 4.68217, 4.673226, 4.719474,
	   4.687384, 4.660241, 4.591836, 4.737024, 4.684212, 4.753448, 4.754027, 4.772588,
	   4.754657, 4.801708, 4.732646, 4.609409, 4.74373, 4.799624, 4.844423, 4.853061,
	   4.780773, 4.812358, 4.718571, 4.924701, 4.895157, 5.024452, 4.940851, 4.893858,
	   4.861413, 5.023933, 4.8769, 4.99654, 4.958747, 4.83442, 4.998678, 4.878566
	};

	std::vector<double> x_data(data.size());
	for (int i = 0; i < x_data.size(); ++i) {
		x_data[i] = i;
	}

	// 데이터에서 최소값과 최대값 찾기
	double offset_initial = *std::min_element(data.begin(), data.end());
	double height_initial = *std::max_element(data.begin(), data.end()) - offset_initial;
	double width_initial = 10.0;  // 폭을 추정하기 위한 일반적인 값
	double center_initial = std::distance(data.begin(), std::max_element(data.begin(), data.end()));

	// Ceres 문제 설정
	Problem problem;
	for (int i = 0; i < data.size(); ++i) {
		CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<GaussianResidual, 1, 1, 1, 1, 1>(
				new GaussianResidual(x_data[i], data[i]));
		problem.AddResidualBlock(cost_function, nullptr, &offset_initial, &height_initial, &width_initial, &center_initial);
	}

	// 솔버 설정 및 최적화 실행
	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	Solver::Summary summary;
	Solve(options, &problem, &summary);

	std::cout << "Final offset: " << offset_initial << std::endl;
	std::cout << "Final height: " << height_initial << std::endl;
	std::cout << "Final width: " << width_initial << std::endl;
	std::cout << "Final center: " << center_initial << std::endl;

	// 결과 파일 생성
	std::string filename = create_filename();
	std::ofstream outFile(filename);
	outFile << "x_value,data,fitted_data\n";
	for (size_t i = 0; i < data.size(); ++i) {
		double fittedValue = offset_initial + height_initial * exp(-pow((x_data[i] - center_initial), 2) / (2 * pow(width_initial, 2)));
		outFile << x_data[i] << "," << data[i] << "," << fittedValue << "\n";
	}
	// 파라미터 결과 추가
	outFile << "Final offset,Final height,Final width,Final center\n";
	outFile << offset_initial << "," << height_initial << "," << width_initial << "," << center_initial << "\n";
	outFile.close();

	cout << "Results saved to " << filename << endl;
}

//#include <Eigen/Dense>
#include ".\\include\eigen3\\Eigen\\Dense"
using namespace Eigen;

// 선형 변환 행렬 계산
Matrix3f calculateLinearTransform(const Vector2f& src1, const Vector2f& src2, const Vector2f& dst1, const Vector2f& dst2) {
	Vector2f srcVector = src2 - src1;
	Vector2f dstVector = dst2 - dst1;

	float scale = dstVector.norm() / srcVector.norm();
	float angle = atan2(dstVector.y(), dstVector.x()) - atan2(srcVector.y(), srcVector.x());

	Matrix3f transform = Matrix3f::Identity();
	transform(0, 0) = cos(angle) * scale;
	transform(0, 1) = -sin(angle) * scale;
	transform(1, 0) = sin(angle) * scale;
	transform(1, 1) = cos(angle) * scale;

	Vector2f translation = dst1 - transform.topLeftCorner<2, 2>() * src1;
	transform(0, 2) = translation.x();
	transform(1, 2) = translation.y();

	return transform;

}

//2점 얼라인먼트는 두 점 사이의 거리와 방향을 기준으로 스케일과 회전을 계산
Matrix3f twoPointAlign(const Vector2f& src1, const Vector2f& src2, const Vector2f& dst1, const Vector2f& dst2) {
	Vector2f srcVector = src2 - src1;
	Vector2f dstVector = dst2 - dst1;

	float scale = dstVector.norm() / srcVector.norm();
	float angle = atan2(dstVector.y(), dstVector.x()) - atan2(srcVector.y(), srcVector.x());

	Matrix3f transform = Matrix3f::Identity();
	transform(0, 0) = cos(angle) * scale;
	transform(0, 1) = -sin(angle) * scale;
	transform(1, 0) = sin(angle) * scale;
	transform(1, 1) = cos(angle) * scale;

	Vector2f translation = dst1 - transform.topLeftCorner<2, 2>() * src1;
	transform(0, 2) = translation.x();
	transform(1, 2) = translation.y();

	return transform;
}

//3점을 사용하여 아핀 변환을 계산합니다. 이 변환은 회전, 이동, 그리고 비율 조정을 포함합니다.
Matrix3f threePointAlign(const vector<Vector2f>& src, const vector<Vector2f>& dst) {
	MatrixXf A(6, 6);
	VectorXf b(6);

	for (int i = 0; i < 3; ++i) {
		A(i * 2, 0) = src[i][0];
		A(i * 2, 1) = src[i][1];
		A(i * 2, 2) = 1;
		A(i * 2, 3) = 0;
		A(i * 2, 4) = 0;
		A(i * 2, 5) = 0;
		b(i * 2) = dst[i][0];

		A(i * 2 + 1, 0) = 0;
		A(i * 2 + 1, 1) = 0;
		A(i * 2 + 1, 2) = 0;
		A(i * 2 + 1, 3) = src[i][0];
		A(i * 2 + 1, 4) = src[i][1];
		A(i * 2 + 1, 5) = 1;
		b(i * 2 + 1) = dst[i][1];
	}

	VectorXf x = A.colPivHouseholderQr().solve(b);
	Matrix3f transform;
	transform << x(0), x(1), x(2),
		x(3), x(4), x(5),
		0, 0, 1;

	return transform;
}

//4점을 사용하여 프로젝티브 변환을 계산합니다. 이는 원근 왜곡을 포함할 수 있는 변환입니다.
Matrix3f fourPointAlign(const vector<Vector2f>& src, const vector<Vector2f>& dst) {
	MatrixXf A(8, 9);
	for (int i = 0; i < 4; i++) {
		A.block<1, 3>(2 * i, 0) = Vector3f(src[i].x(), src[i].y(), 1.0f).transpose();
		A.block<1, 3>(2 * i, 3) = Vector3f(0, 0, 0).transpose();
		A.block<1, 3>(2 * i, 6) = -dst[i].x() * Vector3f(src[i].x(), src[i].y(), 1.0f).transpose();
		A.block<1, 3>(2 * i + 1, 0) = Vector3f(0, 0, 0).transpose();
		A.block<1, 3>(2 * i + 1, 3) = Vector3f(src[i].x(), src[i].y(), 1.0f).transpose();
		A.block<1, 3>(2 * i + 1, 6) = -dst[i].y() * Vector3f(src[i].x(), src[i].y(), 1.0f).transpose();
	}

	JacobiSVD<MatrixXf> svd(A, ComputeFullV);
	VectorXf h = svd.matrixV().col(8);
	Matrix3f transform = Map<Matrix3f>(h.data());

	return transform.transpose();
}

// 좌표 변환 함수
Vector2f transformCoordinate(const Matrix3f& matrix, const Vector2f& point) {
	Vector3f pt(point[0], point[1], 1.0);
	Vector3f transformed_pt = matrix * pt;
	return Vector2f(transformed_pt[0], transformed_pt[1]);
}

/*
x	y
Align1	20	20
Align2	1050	1200
실측치1	1.79 	41.48
실측치2	1070.21 	1182.52

이렇게 2개의 얼라인마크 위치와 실측 값이 있을 경우

x	y
80	190
이 Recipe 좌표로 이동시 실제로 움직여야 할 좌표 위치

변환된 좌표는 약(67.56, 208.95)
*/
void CAFFitting2View::OnBnClickedButtonTestAlignMaterix()
{
	// Initialize source and destination points for the alignment
	Vector2f src1(20, 20), src2(1050, 1200);
	Vector2f dst1(1.79, 41.48), dst2(1070.21, 1182.52);

	// Calculate transformation using two-point alignment
	Matrix3f transform = twoPointAlign(src1, src2, dst1, dst2);

	// Additional source and destination points for three and four point alignment
	vector<Vector2f> srcPoints = { src1, src2, Vector2f(0, 1200), Vector2f(1050, 0) };
	vector<Vector2f> dstPoints = { dst1, dst2, Vector2f(0, 1182.52), Vector2f(1070.21, 41.48) };

	// Uncomment the desired alignment method
	// Matrix3f transform = threePointAlign(srcPoints, dstPoints);
	// Matrix3f transform = fourPointAlign(srcPoints, dstPoints);

	// Define the recipe point and transform it
	Vector2f recipePoint(80, 190);
	Vector2f actualPoint = transformCoordinate(transform, recipePoint);

	cout << "Actual Coordinate: (" << actualPoint[0] << ", " << actualPoint[1] << ")" << endl;
}

// 역변환 함수: 어떤 변환 매트릭스도 역변환 가능
Vector2f inverseTransformCoordinate(const Matrix3f& transform, const Vector2f& transformedPoint) {
	Matrix3f inverseTransform = transform.inverse();  // 역행렬 계산
	Vector3f tp(transformedPoint[0], transformedPoint[1], 1.0);
	Vector3f original_pt = inverseTransform * tp;  // 역변환 적용
	return Vector2f(original_pt[0] / original_pt[2], original_pt[1] / original_pt[2]);  // 동차 좌표에서 일반 좌표로 변환
}

void CAFFitting2View::OnBnClickedButtonTestInverseAlignMaterix()
{
	// 예를 들어 4점 변환 매트릭스 계산
	vector<Vector2f> src = { Vector2f(0, 0), Vector2f(1, 0), Vector2f(0, 1), Vector2f(1, 1) };
	vector<Vector2f> dst = { Vector2f(0, 0), Vector2f(2, 0), Vector2f(0, 2), Vector2f(2, 2) };
	Matrix3f transform = fourPointAlign(src, dst);  // 여기에 사용된 fourPointAlign은 이전 예제에서 정의된 함수

	// 역변환 적용 예제
	Vector2f transformedPoint(1.0, 1.0);
	Vector2f originalPoint = inverseTransformCoordinate(transform, transformedPoint);
	cout << "Original Coordinate: (" << originalPoint[0] << ", " << originalPoint[1] << ")" << endl;
}
/*
int main() {
	// Define the transformation matrix (example values)
	Matrix3f transform;
	transform << 1.0, 0.2, 5.0,  // 임의의 아핀 변환 매트릭스
				-0.2, 0.9, 20.0,
				0.0, 0.0, 1.0;

	// Example transformed point that needs to be converted back to recipe coordinates
	Vector2f transformedPoint(67.56, 208.95);  // 화면에 표시된 모션 좌표

	// Calculate the original coordinates using the inverse transform
	Vector2f originalPoint = inverseTransformCoordinate(transform, transformedPoint);

	cout << "Original Coordinate: (" << originalPoint[0] << ", " << originalPoint[1] << ")"
		 << " - 이 값은 레시피 좌표계로의 역변환 결과입니다." << endl;

	return 0;
}*/