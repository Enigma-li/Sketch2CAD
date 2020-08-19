#ifdef _WITH_GPU_SUPPORT

#pragma once

#define COMPILER_MSVC
#define NOMINMAX

#include <vector>
#include <string>
#include <ostream>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"


class SketchModel {

public:
	SketchModel();
	~SketchModel();

	// load config and preprocessing
	void load_config_and_prebuild_network(std::string& conf_fn, int h, int w);

	// set model index
	void set_network_index(int idx) { model_idx = idx; }

	// set up network
	bool setup_network(int idx, std::vector<int>& ic_nb);
	int get_network_idx() { return model_idx; };

	// warm up the network (forward fake data)
	bool warmup_network();

	// set input tensor
	bool set_input_tensor(std::vector<std::vector<float>>& data);

	// predict output
	bool predict_output(std::vector<int>& nodes_idx, std::vector<std::vector<float>>& net_outputs);

private:
	int model_idx;
	std::string model_dir;									//! model folder
	std::vector<std::string> model_names;							//! model names
	std::vector<std::vector<std::string>> input_node_names;					//! input node names
	std::vector<std::vector<int>> input_node_channels;					//! input node channel number
	std::vector<std::vector<std::string>> output_node_names;				//! output node names
	int iH, iW;																//! input image size

	tensorflow::Status m_status;								//! tensorflow status
	tensorflow::Session* m_session;								//! tensorflow session
	tensorflow::GraphDef m_graph_def;							//! graph definition
	std::vector<std::pair<std::string, tensorflow::Tensor>> m_inputs;			//! input name and tensor pairs
};

#endif
