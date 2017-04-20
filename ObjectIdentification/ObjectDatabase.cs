using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectIdentification
{
    public class ObjectDatabase
    {

        private BFMatcher matcher;

        private double hessianThresh = 300;


        public ObjectDatabase()
        {
            matcher = new BFMatcher(DistanceType.L2);
        }

        public void Train(Image<Gray, byte> img)
        {
            using (UMat uModelImage = img.Mat.GetUMat(Emgu.CV.CvEnum.AccessType.Read))
            {
                
                SURF surfCPU = new SURF(hessianThresh);

                UMat modelDescriptors = new UMat();

                VectorOfKeyPoint modelKeyPoints = new VectorOfKeyPoint();

                surfCPU.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);

                matcher.Add(modelDescriptors);
                

            }
        }


        public void Search(Image<Gray, byte> img)
        {

        }
    }
}
