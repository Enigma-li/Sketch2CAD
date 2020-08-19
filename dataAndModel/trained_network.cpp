#ifdef _WITH_GPU_SUPPORT
#include "trained_network.h"

SketchModel::SketchModel()
{
	m_session = NULL;
	m_inputs.clear();
	model_idx = 0;
	model_names.clear();
	input_node_names.clear();
	input_node_channels.clear();
	output_node_names.clear();
	iH = iW = -1;
}

SketchModel::~SketchModel()
{
	m_session->Close();
	if (m_session) delete m_session;
}

void SketchModel::load_config_and_prebuild_network(std::string & conf_fn, int h, int w)
{
	std::ifstream in(conf_fn);
	if (!in.is_open())
	{
		std::cout << "Error: cannot open input loss file, get: " << conf_fn << std::endl;
		return;
	}
	
	// clear data
	model_names.clear();
	input_node_names.clear();
	input_node_channels.clear();
	output_node_names.clear();

	// set model directory
	model_dir = FileSystem::dir_name(conf_fn);

	iH = h;
	iW = w;

	// load model name
	std::string content;
	while (std::getline(in, content))
	{
		model_names.push_back(content);
	}
	in.close();

	// load per-model node configuration
	std::cout << "\n--Init: load network: " << std::endl;
	for (int mitr = 0; mitr < model_names.size(); mitr++)
	{
		std::cout << "\t" << model_names[mitr] << std::endl;
		std::string per_model_conf_fn = model_dir + "//" + model_names[mitr] + "_node_def.txt";
		std::ifstream m_model_in(per_model_conf_fn);

		if (!m_model_in.is_open())
		{
			std::cout << "Error: cannot load model " << mitr << " configuration, get: " << per_model_conf_fn << std::endl;
			continue;
		}

		std::vector<std::string> cur_inode_names;
		std::vector<int> cur_inode_cnb;
		std::vector<std::string> cur_onode_names;

		std::string content;
		while (std::getline(m_model_in, content))
		{
			std::vector<std::string> sub_strs;
			split_string(content, ' ', sub_strs);

			if (sub_strs[0].compare("Input:") == 0)
			{
				for (int nitr = 1; nitr < sub_strs.size(); nitr++)
				{
					cur_inode_names.push_back(sub_strs[nitr]);
				}
			}
			else if (sub_strs[0].compare("InputChannelNb:") == 0)
			{
				for (int citr = 1; citr < sub_strs.size(); citr++)
				{
					cur_inode_cnb.push_back(std::atoi(sub_strs[citr].c_str()));
				}
			}
			else if (sub_strs[0].compare("Output:") == 0)
			{
				for (int oitr = 1; oitr < sub_strs.size(); oitr++)
				{
					cur_onode_names.push_back(sub_strs[oitr]);
				}
			}
		}
		m_model_in.close();

		input_node_names.push_back(cur_inode_names);
		input_node_channels.push_back(cur_inode_cnb);
		output_node_names.push_back(cur_onode_names);
	}
}

bool SketchModel::setup_network(int idx, std::vector<int>& ic_nb)
{
	std::cout << "\n--Current network: " << model_names[idx] << "\n\tInput channels:";
	// set model index
	model_idx = idx;

	// get input tensor channel size
	ic_nb.clear();
	for (int citr = 0; citr < input_node_channels[model_idx].size(); citr++)
	{
		std::cout << input_node_channels[model_idx][citr] << " ";
		ic_nb.push_back(input_node_channels[model_idx][citr]);
	}
	std::cout << "\n" << std::endl;

	// new session
	if (m_session) delete m_session;
	tensorflow::SessionOptions session_options;
	session_options.config.mutable_gpu_options()->set_allow_growth(true);
	session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);

	m_status = tensorflow::NewSession(/*tensorflow::SessionOptions()*/session_options, &m_session);
	if (!m_status.ok())
	{
		std::cout << "Error: cannot create tensorflow session, get: " << m_status.ToString() << std::endl;
		return false;
	}
	
	std::string model_fn = model_dir + "//" + model_names[model_idx] + "_freezed.pb";
	m_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_fn, &m_graph_def);
	if (!m_status.ok())
	{
		std::cout << "Error: cannot load graph definition, get: " << m_status.ToString() << "\n\tModel file: " << model_fn << std::endl;
		return false;
	}

	m_status = m_session->Create(m_graph_def);
	if (!m_status.ok())
	{
		std::cout << "Error: add graph to current session, get: " << m_status.ToString() << std::endl;
		return false;
	}

	if (!warmup_network())
	{
		return false;
	}

	return true;
}

bool SketchModel::warmup_network()
{
	std::cout << "\n--Warmup network...";
	// set input
	std::vector<std::vector<float>> input_data;
	for (int itr = 0; itr < input_node_channels[model_idx].size(); itr++)
	{
		int dSize = 1 * iH*iW*input_node_channels[model_idx][itr];
		std::vector<float> cur_input_data(dSize, 0.0);
		input_data.push_back(cur_input_data);
	}

	if (!set_input_tensor(input_data))
	{
		return false;
	}

	// predict
	std::vector<std::vector<float>> net_outputs;
	if (!predict_output(std::vector<int>(), net_outputs))
	{
		std::cout << "Error: cannot predict shape, please check tensor filling!!!" << std::endl;
		return false;
	}
	std::cout << "done\n" << std::endl;
	return true;
}

bool SketchModel::set_input_tensor(std::vector<std::vector<float>>& data)
{
	if (data.size() != input_node_names[model_idx].size())
	{
		std::cout << "Error: cannot match data with input node" << std::endl;
		return false;
	}

	m_inputs.clear();
	for (int itr = 0; itr < input_node_names[model_idx].size(); itr++)
	{
		tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, iH, iW, input_node_channels[model_idx][itr] }));
		auto cur_tensor_data_ptr = cur_input.flat<float>().data();
		for (int ditr = 0; ditr < data[itr].size(); ditr++)
		{
			cur_tensor_data_ptr[ditr] = data[itr][ditr];
		}

		m_inputs.push_back(std::pair<std::string, tensorflow::Tensor>(input_node_names[model_idx][itr], cur_input));
	}

	return true;
}

bool SketchModel::predict_output(std::vector<int>& nodes_idx, std::vector<std::vector<float>>& net_outputs)
{
	// forward network
	std::vector<std::string> output_nodes;
	if (nodes_idx.size() == 0)
	{
		output_nodes = output_node_names[model_idx];
	}
	else
	{
		for (int nitr = 0; nitr < nodes_idx.size(); nitr++)
		{
			output_nodes.push_back(output_node_names[model_idx][nitr]);
		}
	}
	
	// run the forward pass and store the resulting tensor to outputs
	std::vector<tensorflow::Tensor> outputs;
	m_status = m_session->Run(m_inputs, output_nodes, {}, &outputs);

	if (!m_status.ok())
	{
		std::cout << "Error: cannot run predictions, get: \n" << m_status.ToString() << std::endl;
		return false;
	}

	// fetch output tensor
	net_outputs.clear();
	for (int opt_itr = 0; opt_itr < output_nodes.size(); opt_itr++)
	{
		tensorflow::Tensor cur_t = outputs[opt_itr];
		std::vector<float> cur_net_out;

		int nb_elts = cur_t.NumElements();
		
		auto tensor_data = cur_t.flat<float>().data();
		for (int d_itr = 0; d_itr < nb_elts; d_itr++)
		{
			cur_net_out.push_back(tensor_data[d_itr]);
		}

		net_outputs.push_back(cur_net_out);
	}

	return true;
}


#endif