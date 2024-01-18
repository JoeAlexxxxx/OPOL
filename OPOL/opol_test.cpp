#include "opol_test.h"

int main(int argc, char** argv)
{
	double mesh_Threadhold = 0.018182;

	double angle_t = 22.5;
	int count_t = 4;

	//参数
	double max_angle_thr = PI / 180.0 * angle_t;
	int max_search_pts = 15;
	int best_count_thr = count_t;
	double intersection_angle_thread = PI / 180.0 * 0;
	double restrict_percent = 0;

	//P25
	std::string mainFolder = R"(E:\Research\LineMatch\P25\Data)";//需要指定的两个文件夹
	std::string outFolder = R"(E:\Research\LineMatch\P25\Test\0103)";
	std::string meshPath = R"(E:\Research\LineMatch\P25\Data\image\Mesh.obj)";
	mesh_Threadhold = 0.018182;

	if (0 != _mkdir(outFolder.c_str()))
	{
		_mkdir(outFolder.c_str());
	}

	//是否使用mesh
	bool withMesh = false;
	//是否输出obj
	bool outObj = true;

	RTMesh mr;
	TriMeshGrid TRGrid;
	if (withMesh)
	{
		//加载mesh
		std::cout << "withMesh" << std::endl;
		int err = vcg::tri::io::Importer<RTMesh>::Open(mr, meshPath.c_str());
		vcg::tri::UpdateBounding<RTMesh>::Box(mr);
		vcg::tri::UpdateNormal<RTMesh>::PerFace(mr);

		TRGrid.Set(mr.face.begin(), mr.face.end(), mr.FN() * 2);
		vcg::tri::UpdateComponentEP<RTMesh>::Set(mr);
	}

	std::vector<std::string> imageNameList;
	std::vector<float> cams_focals;
	std::vector<cv::Mat> camsMat;
	cv::Mat point3D;
	cv::Mat tasksMat;
	int knn_image = 3;

	clock_t start_t, end_t, start_t2;
	double endtime = 0;
	start_t = clock();
	start_t2 = clock();

	//读VisualSFM
	read_VisualSfM(mainFolder, outFolder, imageNameList, cams_focals, camsMat, point3D, tasksMat, knn_image);

	end_t = clock();
	endtime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	std::cout << "read sfm time use = " << endtime << std::endl;
	start_t = clock();

	if (outObj)
	{
		if (0 != _mkdir((outFolder + R"(\obj\)").c_str()))
		{
			_mkdir((outFolder + R"(\obj\)").c_str());
		}
	}

	std::string imageFolder = mainFolder + R"(\image\)";
	//std::string ptsFolder = mainFolder + R"(\pts\)";
	std::string ptsFolder = mainFolder + R"(\pts\)";
	std::string lineFolder = mainFolder + R"(\lines\)";

	if (0 != _mkdir((outFolder + R"(\searchMat\)").c_str()))
	{
		_mkdir((outFolder + R"(\searchMat\)").c_str());
	}

	cv::Mat imsizes_Mf = cv::Mat(imageNameList.size(), 2, CV_32FC1);//记录影像大小
	cv::Mat cameras_Mf = cv::Mat(imageNameList.size(), 12, CV_32FC1);//记录相机参数
	std::vector<SingleImage> single_images; single_images.resize(imageNameList.size());

	std::cout << "start load images..." << std::endl;

	start_t = clock();
	// Stage 1. load all image
	ThreadPool executor{ 16 };
	std::vector<std::future<void>> results;
	for (int i = 0; i < imageNameList.size(); i++)
		results.emplace_back(executor.commit(processSingleImage,
			imageFolder, imageNameList[i], camsMat[i], cams_focals[i], lineFolder, outFolder, &single_images, i, cameras_Mf, imsizes_Mf));
	for (auto& result : results)
		result.get();
	results.clear();

	end_t = clock();
	endtime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	std::cout << "load time use = " << endtime << std::endl;

	start_t = clock();

	ScoreRecorder merge_tool(tasksMat.rows, min_epipolar_ang, min_pairs_num);
	std::cout << "----------------------------------------------------------------------" << std::endl;
	std::cout << "start image pair match..." << std::endl;


	//Stage 2. match pair
	omp_set_nested(0);
#pragma omp parallel for
	for (int i = 0; i < tasksMat.rows; i++)
	{
		if (withMesh)
		{
			pairMatchWithMesh(imageNameList, outFolder, ptsFolder,
				mr, TRGrid, mesh_Threadhold,
				i, tasksMat, single_images, point3D, &merge_tool, i,
				max_search_pts, max_angle_thr, best_count_thr, intersection_angle_thread, restrict_percent,
				outObj);
		}
		else
		{
			pairMatch(imageNameList, outFolder, ptsFolder, i, tasksMat, single_images, point3D,
				&merge_tool, i,
				max_search_pts, max_angle_thr, best_count_thr, intersection_angle_thread, restrict_percent,
				outObj);
		}
	}

	end_t = clock();
	endtime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	std::cout << "match time use = " << endtime << std::endl;
	std::cout << "----------------------------------------------------------------------" << std::endl;
	start_t = clock();

	std::cout << "start merge lines..." << std::endl;
	//Stage 3. merge lines

	merge_lines(imageNameList, outFolder + R"(\)",
		tasksMat, imsizes_Mf,
		cameras_Mf,
		pixel_2_line_dis,
		&merge_tool);
	merge_tool.vote_lines();


	end_t = clock();
	endtime = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	std::cout << "merge time use = " << endtime << std::endl;
	write2obj(outFolder + R"(\)",
		"OPOL.obj",
		merge_tool.ks_store,
		merge_tool.match_ids_store,
		tasksMat.rows);

	std::cout << "resconstruct 3D lines: " << merge_tool.ks_store.size() << std::endl;

	end_t = clock();
	endtime = (double)(end_t - start_t2) / CLOCKS_PER_SEC;
	std::cout << "totally time use = " << endtime << std::endl;


	return 0;
}