/* Copyright 2020, Author Changjian Li. All Rights Reserved.
 * @ Project Sketch2CAD
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "zlib.h"

#include <complex>
#include <vector>
#include <iostream>

namespace tensorflow {

    const int32 input_channel = 6;
    const int32 output_channel = 15;

    REGISTER_OP("DecodeBlock")
            .Input("byte_stream: string")
            .Attr("tensor_size: list(int) >= 3")
            .Output("input_data: float")
            .Output("label_data: float")
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                std::vector<int32> t_size;
                TF_RETURN_IF_ERROR(c->GetAttr("tensor_size", &t_size));
                c->set_output(0, c->MakeShape({t_size[0], t_size[1], input_channel}));
                c->set_output(1, c->MakeShape({t_size[0], t_size[1], output_channel}));
                return Status::OK();
            })
            .Doc(R"doc(The decoder of multi-channel image data block)doc");

    class DecodeBlockOp : public OpKernel {
    public:
        explicit DecodeBlockOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &this->tensor_size));
            OP_REQUIRES(context, this->tensor_size.size() == 3, errors::InvalidArgument("target tensor size must be 3-d, got ", this->tensor_size.size()));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& contents = context->input(0);
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()), errors::InvalidArgument("DecodeBlock expect a scalar, got shape ",
                                                                                                       contents.shape().DebugString()));
            const StringPiece input_bytes = contents.scalar<string>()();

            // allocate the output tensor
            Tensor* input_data_tensor = nullptr;
            std::vector<int64> input_tensor_size;
            input_tensor_size.push_back(this->tensor_size[0]);
            input_tensor_size.push_back(this->tensor_size[1]);
            input_tensor_size.push_back(input_channel);
            TensorShape input_tensor_shape = TensorShape(gtl::ArraySlice<int64>{input_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("input_data", input_tensor_shape, &input_data_tensor));

            Tensor* label_data_tensor = nullptr;
            std::vector<int64> label_tensor_size;
            label_tensor_size.push_back(this->tensor_size[0]);
            label_tensor_size.push_back(this->tensor_size[1]);
            label_tensor_size.push_back(output_channel);
            TensorShape label_tensor_shape = TensorShape(gtl::ArraySlice<int64>{label_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("label_data", label_tensor_shape, &label_data_tensor));

            // assemble data into tensor
            auto input_data_ptr = input_data_tensor->flat<float>();
            auto label_data_ptr = label_data_tensor->flat<float>();

            // uncompress the byte stream
            int out_data_size = -1;
            float* inflate_data = inflation_byte(input_bytes, out_data_size);
            OP_REQUIRES(context, out_data_size > 0, errors::InvalidArgument("Zlib inflation error, got size: ", out_data_size));
            OP_REQUIRES(context, (out_data_size - (int)(this->tensor_size[0]*this->tensor_size[1]*this->tensor_size[2])) == 0, errors::InvalidArgument("Inflated data mismatch, got ", out_data_size));

//            // without compression
//            int out_data_size = -1;
//            float* inflate_data = (float *)input_bytes.data();
//            out_data_size = (int)input_bytes.size();
//            out_data_size /= 4;
//            OP_REQUIRES(context, (out_data_size - (int)(this->tensor_size[0]*this->tensor_size[1]*this->tensor_size[2])) == 0, errors::InvalidArgument("Data mismatch, got ", out_data_size));


            // set tensor value
            int64 height = this->tensor_size[0];
            int64 width = this->tensor_size[1];
            int64 channel = this->tensor_size[2];

            float dmin = 100000.0, dmax = -100000.0;
            for(int ritr=0; ritr<height; ritr++)
            {
                for(int citr=0; citr<width; citr++)
                {
                    int64 idx = ritr*width + citr;

                    input_data_ptr(idx*input_channel+0) = inflate_data[idx*channel+0];             // user stroke
                    input_data_ptr(idx*input_channel+1) = inflate_data[idx*channel+1];             // scaffold lines
                    input_data_ptr(idx*input_channel+2) = inflate_data[idx*channel+2];             // context normal x
                    input_data_ptr(idx*input_channel+3) = inflate_data[idx*channel+3];             // context normal y
                    input_data_ptr(idx*input_channel+4) = inflate_data[idx*channel+4];             // context normal z
                    input_data_ptr(idx*input_channel+5) = inflate_data[idx*channel+5];             // context depth

                    label_data_ptr(idx*output_channel+0) = inflate_data[idx*channel+6];            // stitching face
                    label_data_ptr(idx*output_channel+1) = inflate_data[idx*channel+7];            // base curve
                    label_data_ptr(idx*output_channel+2) = inflate_data[idx*channel+8];            // profile curve
                    label_data_ptr(idx*output_channel+3) = inflate_data[idx*channel+9];            // offset curve
                    label_data_ptr(idx*output_channel+4) = inflate_data[idx*channel+10];           // shape mask
                    label_data_ptr(idx*output_channel+5) = inflate_data[idx*channel+11];           // offset distance
                    label_data_ptr(idx*output_channel+6) = inflate_data[idx*channel+12];           // offset direction x
                    label_data_ptr(idx*output_channel+7) = inflate_data[idx*channel+13];           // offset direction y
                    label_data_ptr(idx*output_channel+8) = inflate_data[idx*channel+14];           // offset direction z
                    label_data_ptr(idx*output_channel+9) = inflate_data[idx*channel+15];           // offset sign
                    label_data_ptr(idx*output_channel+10) = inflate_data[idx*channel+16];          // operation type

                    // curve classification label
                    label_data_ptr(idx*output_channel+11) = inflate_data[idx*channel+7];                // base
                    label_data_ptr(idx*output_channel+12) = inflate_data[idx*channel+9];                // offset
                    if(inflate_data[idx*channel+7] > 0)             // check conflict
                    {
                        label_data_ptr(idx*output_channel+12) = 0.0;
                    }
                    label_data_ptr(idx*output_channel+13) = inflate_data[idx*channel+8];                // profile
                    if(inflate_data[idx*channel+7] > 0 || inflate_data[idx*channel+9] > 0)      // check conflict
                    {
                        label_data_ptr(idx*output_channel+13) = 0.0;
                    }

                    // curve regression label
                    if(inflate_data[idx*channel+7] > 0)
                    {
                        label_data_ptr(idx*output_channel+14) = 0.0;
                    }
                    else if(inflate_data[idx*channel+9] > 0)
                    {
                        label_data_ptr(idx*output_channel+14) = 1.0;
                    }
                    else if(inflate_data[idx*channel+8] > 0)
                    {
                        label_data_ptr(idx*output_channel+14) = 2.0;
                    }
                    else
                    {
                        label_data_ptr(idx*output_channel+14) = 0.0;
                    }
                }
            }

            delete[] inflate_data;
        }

    private:
        float* inflation_byte(const StringPiece &input_bytes, int& out_size)
        {
            // zipper stream
            z_stream infstream;
            infstream.zalloc = Z_NULL;
            infstream.zfree = Z_NULL;
            infstream.opaque = Z_NULL;

            // set input, output
            Byte* uncompressed_data = new Byte[100000000];
            //  delete it outside

            infstream.avail_in = (uInt)input_bytes.size();
            infstream.next_in = (Bytef*)input_bytes.data();
            infstream.avail_out = (uLong)100000000;
            infstream.next_out = uncompressed_data;

            // uncompress work
            int nErr, real_out_size = -1;

            nErr = inflateInit(&infstream);
            if(nErr != Z_OK)
            {
                out_size = -1;
                return nullptr;
            }
            nErr = inflate(&infstream, Z_FINISH);
            if(nErr == Z_STREAM_END)
            {
                real_out_size = (int)infstream.total_out;
            }
            inflateEnd(&infstream);

            // assign data
            real_out_size /= 4;
            out_size = real_out_size;

            return (float *)uncompressed_data;
        }

    private:
        std::vector<int64> tensor_size;
    };

    REGISTER_KERNEL_BUILDER(Name("DecodeBlock").Device(DEVICE_CPU), DecodeBlockOp);

}
