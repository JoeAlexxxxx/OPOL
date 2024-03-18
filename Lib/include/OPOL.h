#define GLSM
#ifdef GLSM
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <fstream>
#include <Eigen/Dense>
//#include <cuda_runtime.h>
//#include <cusolverDn.h>
#include <vector>
#include <queue>
#include <atomic>
#include <future>
#include <condition_variable>
#include <thread>
#include <functional>
#include <stdexcept>
#include <omp.h>
#include <numeric> 
#include <direct.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include "hnswlib.h"
#include "LSD.hpp"
#include "knncuda.h"
#include "plylib.h"

#include <vcg/complex/complex.h>
#include <vcg/simplex/face/component_ep.h>
#include <vcg/complex/algorithms/update/component_ep.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_ply.h>

#define  THREADPOOL_MAX_NUM 16
class ThreadPool
{
	using Task = std::function<void()>;
	std::vector<std::thread> _pool;
	std::queue<Task> _tasks;
	std::mutex _lock;
	std::condition_variable _task_cv;
	std::atomic<bool> _stop{ false };

public:
	inline ThreadPool(unsigned short size = 4) { addThread(size); }
	inline ~ThreadPool()
	{
		_stop.store(true);
		_task_cv.notify_all();
		for (std::thread& thread : _pool) {
			if (thread.joinable())
				thread.join();
		}
	}

public:

	template<class F, class... Args>
	auto commit(F&& f, Args&&... args) ->std::future<decltype(f(args...))>
	{
		if (_stop.load())
			throw std::runtime_error("commit on ThreadPool is stopped.");

		using RetType = decltype(f(args...)); // typename std::result_of<F(Args...)>::type, 
		auto task = std::make_shared<std::packaged_task<RetType()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
			);
		std::future<RetType> future = task->get_future();
		{
			std::lock_guard<std::mutex> lock{ _lock };
			_tasks.emplace([task]() {
				(*task)();
				});
		}

		_task_cv.notify_one();

		return future;
	}

private:
	void addThread(unsigned short size)
	{
		for (; _pool.size() < THREADPOOL_MAX_NUM && size > 0; --size)
		{
			_pool.emplace_back([this]
				{
					while (!_stop.load())
					{
						Task task;

						{
							std::unique_lock<std::mutex> lock{ this->_lock };
							this->_task_cv.wait(lock, [this] {
								return this->_stop.load() || !this->_tasks.empty();
								});
							if (this->_stop.load() && this->_tasks.empty())
								return;
							task = move(this->_tasks.front());
							_tasks.pop();
						}
						task();
					}
				});
		}
	}
};

#pragma region ScoreRecorder
struct pair_m
{
	int l3d_i, l3d_j;
	float score;
};

struct TASK
{
	unsigned int l3d_size;
	std::vector<unsigned int> ori_ID;
	std::vector<float>epipolar_ang;
	std::vector<int>consistent_size;
	std::vector<float> scores;
	std::vector<size_t> sort_ids;

	unsigned int sort_id_add = 0;

	std::vector< std::vector<unsigned int>> dst_k;
	std::vector< std::vector<unsigned int>> dst_i;

	bool valid_epipolar(float minmum_ang)
	{
		return epipolar_ang[sort_ids[sort_id_add]] > minmum_ang;
	}

	bool valid_aligned(int minmum_align)
	{
		return consistent_size[sort_ids[sort_id_add]] >= minmum_align;
	}

	float max_score()
	{
		float score_;
		while (sort_id_add < l3d_size)
		{
			score_ = scores[sort_ids[sort_id_add]];
			if (score_ == 0)
				sort_id_add++;
			else
				return score_;
		}

		return 0;
	}

	unsigned int max_id()
	{
		return sort_ids[sort_id_add];
	}

	unsigned int max_ori_id()
	{
		return ori_ID[sort_ids[sort_id_add]];
	}

	void next_score()
	{
		scores[sort_ids[sort_id_add]] = 0;
		sort_id_add++;
	}

	unsigned int ori_id(unsigned int l3d_id)
	{
		return ori_ID[sort_ids[sort_id_add]];
	}

};

class ScoreRecorder
{
	unsigned int sum_l3d_size = 0;

	std::vector<TASK> task_vec;

	void release_conflicts(unsigned int stereo_id, unsigned int l3d_id);

	void max_across_view();

	float max_score = 0;

	float current_score = 0;

	unsigned int max_stereo = 0;

	float minmum_epipolar;

	int minmum_align;

public:

	std::vector<unsigned int> ks_store;

	std::vector<unsigned int> match_ids_store;

	int get_l3dSize(int task_id);

	ScoreRecorder(int task_size, float minmum_epipolar_, int minmum_align_);

	void initial_tasks(int task_id, int l3d_size);

	void add_ori_id(int task_id, int l3d_id, int ori_id, float ep_ang);

	void add_connections(int stereo_i, int stereo_j, std::vector<pair_m>* l3d_pair_vec);

	void vote_lines();
};
#pragma endregion

#pragma region mesh_depend

class BaseVertex;
class BaseEdge;
class BaseFace;

struct BaseUsedTypes : public vcg::UsedTypes<vcg::Use<BaseVertex>::AsVertexType, vcg::Use<BaseEdge>::AsEdgeType, vcg::Use<BaseFace>::AsFaceType> {};

class BaseVertex : public vcg::Vertex< BaseUsedTypes,
	vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags  > {};

class BaseEdge : public vcg::Edge< BaseUsedTypes> {};

class BaseFace : public vcg::Face< BaseUsedTypes,
	vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::BitFlags, vcg::face::Mark, vcg::face::EmptyEdgePlane > {};

class BaseMesh : public vcg::tri::TriMesh<std::vector<BaseVertex>, std::vector<BaseFace> > {};


class RTVertex;
class RTEdge;
class RTFace;

struct RTUsedTypes : public vcg::UsedTypes<vcg::Use<RTVertex>::AsVertexType, vcg::Use<RTEdge>::AsEdgeType, vcg::Use<RTFace>::AsFaceType> {};

class RTVertex : public vcg::Vertex< RTUsedTypes,
	vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags  > {};

class RTEdge : public vcg::Edge< RTUsedTypes> {};

class RTFace : public vcg::Face< RTUsedTypes,
	vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::EdgePlane, vcg::face::Mark, vcg::face::BitFlags > {};

class RTMesh : public vcg::tri::TriMesh<std::vector<RTVertex>, std::vector<RTFace> > {};

typedef typename RTMesh::ScalarType ScalarType;
typedef typename RTMesh::CoordType CoordType;
typedef typename RTMesh::FaceType FaceType;
typedef vcg::GridStaticPtr<FaceType, ScalarType> TriMeshGrid;

#pragma endregion


BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
	namespace serialization {

		/** Serialization support for cv::Mat */
		template<class Archive>
		void save(Archive& ar, const cv::Mat& m, const unsigned int version)
		{
			size_t elem_size = m.elemSize();
			size_t elem_type = m.type();

			ar& m.cols;
			ar& m.rows;
			ar& elem_size;
			ar& elem_type;

			const size_t data_size = m.cols * m.rows * elem_size;
			ar& boost::serialization::make_array(m.ptr(), data_size);
		}

		/** Serialization support for cv::Mat */
		template <class Archive>
		void load(Archive& ar, cv::Mat& m, const unsigned int version)
		{
			int cols, rows;
			size_t elem_size, elem_type;

			ar& cols;
			ar& rows;
			ar& elem_size;
			ar& elem_type;

			m.create(rows, cols, elem_type);

			size_t data_size = m.cols * m.rows * elem_size;
			ar& boost::serialization::make_array(m.ptr(), data_size);
		}

	}
}

//struct PointsPair
//{
//	cv::Point2d left_point;
//	cv::Point2d right_point;
//	cv::Point3d space_point;
//};
#define PI 3.14159265359
#define PI_T2 6.28318530718000//3.14159265359*2

struct LineMatch
{
	int left_indx;
	int right_indx;
	LineMatch() : left_indx(-1), right_indx(-1) {}
	LineMatch(int l, int r) : left_indx(l), right_indx(r) {}
};

struct PointUse
{
	int use_count = 0;
	int indx_offset = 0;
};
struct IDX_Point
{
	cv::Point2d pt;
	int indx;
	double ori;
};

struct SingleImage
{
	//std::vector<PointsPair> pts_pair;
	//cv::Mat i_points;
	std::vector<IDX_Point> i_points;
	cv::Mat i_image;
	cv::Mat i_lines;
	cv::Mat i_linesF;
	cv::Mat i_cam;
	cv::Mat i_line_middle_pts;
	cv::Mat i_middle_near_pts_idx;
	cv::Mat i_start_near_pts_idx;
	cv::Mat i_end_near_pts_idx;
	cv::Mat l_search_map;
	int image_width;
	int image_height;
};

struct MatchPair
{
	cv::Mat F;
	cv::Mat AE;
	cv::Mat C; // 相机中心
	cv::Mat M;
	cv::Mat p4;
	cv::Mat e;
	cv::Mat A;
	cv::Mat principal_r;
	double norm_3;

	//std::vector<PointsPair> pts_pair;
	cv::Mat left_points;
	cv::Mat left_image;
	cv::Mat left_lines;
	cv::Mat left_linesF;
	cv::Mat left_cam;
	cv::Mat left_line_middle_pts;
	cv::Mat left_middle_near_pts_idx;
	cv::Mat left_start_near_pts_idx;
	cv::Mat left_end_near_pts_idx;

	cv::Mat right_points;
	cv::Mat right_image;
	cv::Mat right_lines;
	cv::Mat right_linesF;
	cv::Mat right_cam;

	cv::Mat l_search_map;
	cv::Mat spacePoints;
	cv::Mat spacePointsIdx;
	cv::Mat orientation;

	std::string outObjName;

	int left_width;
	int left_height;
	int right_width;
	int right_height;

};

struct LineMatchResult
{
	int left_line_indx;
	int right_line_indx;
	int point_indx;
	double angle_change;
	double angle_change_src;

	LineMatchResult() :left_line_indx(-1), right_line_indx(-1), point_indx(-1), angle_change(-9999), angle_change_src(-99999) {}
	LineMatchResult(int a, int b, int c, double d, double e) :left_line_indx(a), right_line_indx(b), point_indx(c), angle_change(d), angle_change_src(e) {}
};

struct LineMatchWeight
{
	int left_line_indx;
	double match_wight;

	double startX = 0, startY = 0, startZ = 0;
	double endX = 0, endY = 0, endZ = 0;
	double ep_angle = 0;

	LineMatchWeight() :left_line_indx(-1), match_wight(0) {}
	LineMatchWeight(int a, double d) :left_line_indx(a), match_wight(d) {}
};

struct EigenValue
{
	double value;
	int indx;
	EigenValue() : value(0), indx(-1) {}
	EigenValue(double a, int b) : value(a), indx(b) {}

};

struct AVector
{
	int row_index;
	int col_index;
	float value;
	AVector() : row_index(-1), col_index(-1), value(0) {}
	AVector(int r, int c, float v) : row_index(r), col_index(c), value(v) {}
};

struct LineMatchSelect
{
	int left_line_indx;
	int right_line_indx;

	LineMatchSelect() :left_line_indx(-1), right_line_indx(-1) {}
	LineMatchSelect(int a, int b) :left_line_indx(a), right_line_indx(b) {}
};



//线匹配主函数
//void GenLineMatch(std::string left_image, std::string left_cam, std::string left_line, std::string left_pts,
//	std::string right_image, std::string right_cam, std::string right_line, std::string right_pts,
//	std::string out_obj_name);

//多线程
//void multiMatchMethod(std::string imageFolder, std::string lineFolder, std::string outFolder,
//	std::vector<std::string> imageNameList, std::vector<int> taskListPath1, std::vector<int> taskListPath2, int index);
//void multiMatchMethod(std::vector<MatchPair>& mp_vec, int thread_indx);

//1. 读取文件
//读取线文件
void loadLines(std::string left_lines_path, std::string right_lines_path, MatchPair& mp);

//读取点文件
void loadPoints(std::string left_pts_path, std::string right_pts_path, MatchPair& mp);
void loadPoints(std::string left_pts_path, std::string right_pts_path, MatchPair& mp, int XOffset, int YOffset, int XOffset_r, int YOffset_r);

//读取相机矩阵
void loadCams(std::string left_cams_path, std::string right_cams_path, MatchPair& mp);

//读取影像
void loadImage(std::string left_img_path, std::string right_img_path, MatchPair& mp);

//线提取
void lineDetect(cv::Mat img, SingleImage& ip);
void lineDetect(cv::Mat img, SingleImage& ip, std::string outPath);//使用opencv提取
void lineDetect_LSD(cv::Mat img, SingleImage& ip, std::string outPath);//使用LSD提取线

void loadLineFromFile(std::string lineFilePath, SingleImage& ip, std::string outPath);//从txt文件读取线


//2.基础函数
//计算特征点主方向

//通过点线构建单应矩阵
cv::Mat getHMatrixFromPointLine(cv::Mat left_line, cv::Mat right_line, cv::Mat left_point,
	cv::Mat right_point, cv::Mat epipole_point, cv::Mat f_matrix);
void getHMatrixFromPointLine(cv::Mat left_line, cv::Mat right_line, cv::Mat left_point,
	cv::Mat right_point, cv::Mat epipole_point, cv::Mat f_matrix, float* H_M);
void getHMatrixFromPointLine(cv::Mat left_line, cv::Mat right_line, float* left_point,
	float* right_point, cv::Mat epipole_point, cv::Mat f_matrix, float* H_M);

//通过点线构建单应矩阵
void getHMatrixFrom3Pts1Line(float* pts1, float* pts2, float* lines1, float* lines2F, double* H_Matrix);

//构建三维空间线
void get3DLine(cv::Mat CM, cv::Mat CN, cv::Mat F, cv::Mat l1, cv::Mat l2, cv::Mat& pt3d, double& angle_c);
void get3DLine(cv::Mat CM, cv::Mat CN, cv::Mat F, cv::Mat l1, cv::Mat l2, cv::Mat& pt3d);

//通过构建一维搜索序列
void getSearchVec(MatchPair mp, cv::Mat middle_pt, double min_depth, double max_depth, std::vector<int>& vec_xx, std::vector<int>& vec_yy);

//寻找临近点
void searchNearPoint(cv::Mat queriesMat, cv::Mat searchMat, cv::Mat& vecIndx, int queryNum);
void multiSearchNearPoint(MatchPair& mp, int queryNum, int threadIndx);
void allSearchNearPoint(MatchPair& mp, int queryNum);
void hnswSearchNearPoint(MatchPair& mp, int queryNum);
void cudaSearchNearPoint(MatchPair& mp, int queryNum);
//knnCuda，仅搜索中间点
void cudaSearchMiddleNearPoint(MatchPair& mp, int queryNum);
//void searchNearPoint(cv::Mat queriesMat, cv::Mat searchMat, std::vector<int>& vecIndx, std::vector<float>& vecDist, int queryNum);

//构建搜索图
void createMap(int imr, int imc, cv::Mat lines_Mf, cv::Mat& inter_map_);

//线段经过的点
void Bresenham(int x1, int y1, int const x2, int const y2, std::vector<int>& xx, std::vector<int>& yy);

//像点前方交会
void triangulate(cv::Mat left_cam, cv::Mat right_cam, cv::Mat left_pts, cv::Mat right_pts, cv::Mat& space_pts);

//从所有空间点的Mat中获取三位点
void getSpacePointMat(cv::Mat allSpacePoints, cv::Mat spaceIndx, cv::Mat& spacePointsMat);

//相机矩阵计算基础矩阵
void computeFAE(cv::Mat P1, cv::Mat P2, cv::Mat& F_Mf, cv::Mat& AE_Mf);

//斜对角矩阵
void skew_mat3(cv::Mat M1f, cv::Mat& skew_Mf);

//根据相机矩阵初始化相关参数
void iniCamMatrix(MatchPair& mp);

//根据物方点计算深度范围
void minMaxDepth(cv::Mat spacepoints, cv::Mat search_indx, cv::Mat cam_matrix, double norm_3, double& mindepth, double& maxdepth);

//计算深度范围对应的影像像素范围
//void depthRange(float* C, float* CM, float* CN, float* itpm, float* principal_r, float mindepth, float maxdepth,
//	float& x1, float& y1, float& x2, float& y2);

//给特征值排序
bool cmp(EigenValue a, EigenValue b);
bool cmp_Resultmatch(LineMatchResult a, LineMatchResult b);

void calcu_dis_object(RTMesh& mr, TriMeshGrid TRGrid);

//基本计算
float norm_v3(float* v3);
void mult_3_4_4(float* CM, float* x, float* res_3_1);
void mult_3_3_1(float* o, float* e, float* res_3_1);
void norm_by_v3(float* v);
float norm_v2(float* v2);
void cross_v3(float* v1, float* v2, float* v3);
float dot_v3(float* v1, float* v2);
float cos_vec3(float* veci, float* vecj);
float point_2_line_dis(float* pt, float* linef);
float dot_v2(float x1, float y1, float x2, float y2);
bool twoLines_intersec(float* pt1, float* pt2, float* tl1, float* tl2, float intersecratio);
bool ID_in_array(std::vector<short> ids, int num);

//读取单张影像
void loadSingleImage(std::string left_img_path, SingleImage& ip);

//读取相机矩阵
void loadSingleCam(std::string left_cams_path, SingleImage& ip);

//读取线文件
void loadSingleLines(std::string left_lines_path, SingleImage& ip);

//串点
void getPointsLink(SingleImage lp, SingleImage rp, MatchPair& mp);
void getPointsLink(std::string ptsLinkFile, MatchPair& mp);

//读取点
void loadSinglePoints(std::string left_pts_path, SingleImage& ip,
	int XOffset, int YOffset);
void loadSinglePoints(std::string left_pts_path, SingleImage& ip);

void GenLineMatch(MatchPair& mp, int max_search_pts, double max_angle_thr, double best_count_thr,
	double intersection_angle_thread = 0, double restrict_precent = 1.0,
	bool out_obj = true, std::string out_obj_name = "");

void GenLineMatch(MatchPair& mp, int max_search_pts, double max_angle_thr, double best_count_thr,
	ScoreRecorder& merge_tool, int task_id,
	double intersection_angle_thread = 0, double restrict_precent = 1.0,
	bool out_obj = true, std::string out_obj_name = "", std::string out_dir = "");

void GenLineMatchwithMesh(MatchPair& mp, int max_search_pts, double max_angle_thr, double best_count_thr,
	ScoreRecorder& merge_tool, int task_id,
	RTMesh& mr, TriMeshGrid TRGrid, double mesh_th,
	double intersection_angle_thread = 0, double restrict_precent = 1.0,
	bool out_obj = true, std::string out_obj_name = "", std::string out_dir = "");

double angleDiff(double a, double b);

void processSingleImage(std::string image_folder, std::string image_name, cv::Mat camRow, float camFocal,
	std::string line_folder, std::string outfolder, std::vector<SingleImage>* si, int thread_num,
	cv::Mat& camsAll, cv::Mat& imageSizes);

//void processSingleImage(std::string image_folder, std::string image_name, std::string cam_folder, SingleImage& si,
//	std::string lines_folder, std::string line_name);

//匹配
void pairMatch(std::vector<std::string> imageNameList, std::string outFolder, std::string ptsFolder, int i,
	cv::Mat taskMat, std::vector<SingleImage> single_images, cv::Mat allSpacePoints,
	ScoreRecorder* merge_tool, int task_id,
	int max_search_pts, double max_angle_thr, double best_count_thr,
	double intersection_angle_thread = 0, double restrict_precent = 1.0,
	bool outObj = false);

void pairMatchWithMesh(std::vector<std::string> imageNameList, std::string outFolder, std::string ptsFolder,
	RTMesh& mr, TriMeshGrid TRGrid, double mesh_th,
	int i,
	cv::Mat taskMat, std::vector<SingleImage> single_images, cv::Mat allSpacePoints,
	ScoreRecorder* merge_tool, int task_id,
	int max_search_pts, double max_angle_thr, double best_count_thr,
	double intersection_angle_thread = 0, double restrict_precent = 1.0,
	bool outObj = false);

void MultiPairMatch(int threadId, std::vector<std::string> imageNameList, std::string outFolder, std::string ptsFolder,
	std::vector<int> taskListPath1, std::vector<int> taskListPath2, std::vector<SingleImage> single_images,
	ScoreRecorder* merge_tool,
	int max_search_pts, double max_angle_thr, double best_count_thr,
	double intersection_angle_thread = 0, double restrict_precent = 1.0);

void write2obj(std::string input_folder, std::string outname, std::vector<unsigned int>& ks,
	std::vector<unsigned int>& ids, int pairsize);
void saveMat(std::string save_name, cv::Mat m);
void readMat(std::string name, cv::Mat& mat);


#ifndef maximum_ang
#define maximum_ang 0.0872664625997 
#endif 

#ifndef line_2_line_intersec
#define line_2_line_intersec 0.5
#endif 

#ifndef max_map_size
#define max_map_size 100
#endif 

#ifndef max_line3D_size
#define max_line3D_size 200
#endif 

#ifndef max_line2D_size
#define max_line2D_size 200
#endif 

#ifndef min_epipolar_ang
#define min_epipolar_ang 0.087266
#endif 

#ifndef min_pairs_num
#define min_pairs_num 2
#endif 

#ifndef pixel_2_line_dis
#define pixel_2_line_dis 2
#endif 

void merge_lines(std::vector<std::string>image_names, std::string input_folder,
	cv::Mat imidx_Mf, cv::Mat imsizes_Mf,
	cv::Mat cameras_Mf,
	float dist,
	ScoreRecorder* merge_tool);

int read_VisualSfM(std::string inputFolder, std::string outFolder,
	std::vector<std::string>& image_names,
	std::vector<float>& cams_focals,
	std::vector<cv::Mat>& cams_RT,
	cv::Mat& points_space3D,
	cv::Mat& imidx_Mf_,
	int knn_image);

#endif // GLSM
