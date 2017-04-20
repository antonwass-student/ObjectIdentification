using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.Util;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.XFeatures2D;
using System.Diagnostics;
using Emgu.CV.Util;
using Emgu.CV.Features2D;

namespace ObjectIdentification
{
    public partial class Form1 : Form
    {
        Image<Gray, byte> model, observed;


        public Form1()
        {
            InitializeComponent();
        }

        private void loadToolStripMenuItem_Click(object sender, EventArgs e)
        {
            
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(pictureBox1.Image != null)
            {
                pictureBox1.Image.RotateFlip(RotateFlipType.Rotate90FlipNone);
                pictureBox1.Invalidate();
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                pictureBox1.Image.RotateFlip(RotateFlipType.Rotate270FlipNone);
                pictureBox1.Invalidate();
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Debug.WriteLine("asdf");
            int k = 2;
            double uniquenessThreshold = 0.8;
            double hessianThresh = 300;

            Stopwatch watch;
            Mat homography;
            Mat mask;

            VectorOfKeyPoint modelKeyPoints = new VectorOfKeyPoint();
            VectorOfKeyPoint observedKeyPoints = new VectorOfKeyPoint();
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();

            Mat observedImage = new Mat();
            Mat modelImage = new Mat();
            

            observedImage = observed.Mat;
            modelImage = model.Mat;

            Debug.WriteLine("ddd");

            using (UMat uModelImage = modelImage.GetUMat(Emgu.CV.CvEnum.AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(Emgu.CV.CvEnum.AccessType.Read))
            {
                SURF surfCPU = new SURF(hessianThresh);

                UMat modelDescriptors = new UMat();

                surfCPU.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);

                Debug.WriteLine("Starting");
                watch = Stopwatch.StartNew();

                UMat observedDescriptors = new UMat();
                surfCPU.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);

                BFMatcher matcher = new BFMatcher(DistanceType.L2);
                matcher.Add(modelDescriptors);

                matcher.KnnMatch(observedDescriptors, matches, k, null);

                mask = new Mat(matches.Size, 1, Emgu.CV.CvEnum.DepthType.Cv8U, 1);
                mask.SetTo(new MCvScalar(255));
                Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                int nonZeroCount = CvInvoke.CountNonZero(mask);
                if(nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, matches, mask, 1.5, 20);
                    if(nonZeroCount >= 4)
                    {
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, matches, mask, 2);
                    }
                }

                watch.Stop();

            }

            Debug.WriteLine("Match complete: " + watch.ElapsedMilliseconds);



            MKeyPoint[] points = modelKeyPoints.ToArray();

            Bitmap newBitmap = new Bitmap(pictureBox2.Image.Width, pictureBox2.Image.Height);

            using(Graphics g = Graphics.FromImage(newBitmap))
            {
                g.DrawImage(pictureBox2.Image, 0, 0);
                foreach (MKeyPoint p in points)
                {
                    g.DrawEllipse(Pens.Green, p.Point.X, p.Point.Y, 5, 5);
                }
                
            }
            pictureBox2.Image = newBitmap;
            pictureBox2.Invalidate();


        }



        private void cameraToolStripMenuItem_Click(object sender, EventArgs e)
        {
            VideoCapture capture = new VideoCapture();

            pictureBox1.Image = capture.QueryFrame().Bitmap;
            pictureBox1.Invalidate();
        }

        private void modelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            //set model
            string path;
            OpenFileDialog file = new OpenFileDialog();
            if (file.ShowDialog() == DialogResult.OK)
            {
                Image<Bgr, Byte> my_Image = new Image<Bgr, byte>(file.FileName);


                model = my_Image.Convert<Gray, byte>();

                pictureBox2.Image = model.ToBitmap();

                pictureBox2.Invalidate();
            }
        }

        private void observedToolStripMenuItem_Click(object sender, EventArgs e)
        {
            //set observed
            string path;
            OpenFileDialog file = new OpenFileDialog();
            if (file.ShowDialog() == DialogResult.OK)
            {
                Image<Bgr, Byte> my_Image = new Image<Bgr, byte>(file.FileName);


                observed = my_Image.Convert<Gray, byte>();

                pictureBox1.Image = observed.ToBitmap();

                pictureBox1.Invalidate();
            }
        }
    }
}
