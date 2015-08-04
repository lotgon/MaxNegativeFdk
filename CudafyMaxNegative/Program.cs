using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using System.IO;

using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Diagnostics;


namespace CudafyMaxNegative
{
    class Program
    {
        static void Main(string[] args)
        {
            List<float> pricesList = new List<float>();
            List<string> datesList = new List<string>();

            using( var streamReader = new System.IO.StreamReader(Settings1.Default.InputCsvFileName) )
            {
                CsvReader csv = new CsvReader(streamReader);
                csv.Configuration.HasHeaderRecord = true;
                while (csv.Read())
                {
                    string header = csv.GetField<string>(0);
                    float price = csv.GetField<float>(1);
                    string dataTime = csv.GetField<string>(2);

                    pricesList.Add(price);
                    datesList.Add(dataTime);
                }
            }

            using( StreamWriter sw = new StreamWriter(Settings1.Default.InputCsvFileName+"_Cudafy.csv"))
            {
                sw.WriteLine("Date, Tp, Drawdown,BarDuration");
                MeasureTime("GPGPU seq", 1, ()=>
                    {
                        CalculateAll(null, pricesList.ToArray(), datesList.ToArray(), 0.5f, 2, 0.1f);
                    });
            }

        }

        static GPGPU InitGPU()
        {
            CudafyModes.Target = eGPUType.OpenCL; // To use OpenCL, change this enum
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

            int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            if (deviceCount == 0)
            {
                Console.WriteLine("No suitable {0} devices found.", CudafyModes.Target);
            }
            CudafyModes.DeviceId = deviceCount-1;
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            Console.WriteLine("Running examples using {0}", gpu.GetDeviceProperties(false).Name);

            //CudafyTranslator.GenerateDebug = true;
            CudafyModule km = CudafyTranslator.Cudafy();
            gpu.LoadModule(km);

            return gpu;
        }


        static void CalculateAll(StreamWriter sq, float[] prices, string[] dates, float startTp, float endTp, float stepTp)
        {
            GPGPU gpu = InitGPU();

            int N = prices.Length;
            float[] drawdown = new float[N];
            int[] duration = new int[N];

            float[] dev_prices = gpu.Allocate<float>(prices);
            float[] dev_drawdown = gpu.Allocate<float>(drawdown);
            int[] dev_duration= gpu.Allocate<int>(duration);

            gpu.CopyToDevice(prices, dev_prices);

            for (float currTp = startTp; currTp <= endTp; currTp += stepTp)
            {
                int BLOCK_SIZE = 32;
                gpu.Launch((N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE).calculate(dev_prices, currTp, N, dev_drawdown, dev_duration);

                gpu.CopyFromDevice<float>(dev_drawdown, drawdown);
                gpu.CopyFromDevice<int>(dev_duration, duration);

                if( sq!= null)
                {
                    for( int i=0;i<N;i++)
                    {
                        sq.WriteLine(string.Format("{0}, {1}, {2}, {3}", dates[i], currTp, Math.Round(drawdown[i], 6), duration[i]));
                    }
                }
            }
            gpu.FreeAll();
        }
        [Cudafy]
        public static void calculate(GThread gt, float[] prices, float currTp, int N, float[] drawdownArray, int[] durationArray)
        {
            int absTx = gt.blockIdx.x * gt.blockDim.x + gt.threadIdx.x;

            if (absTx >= N)
                return;

            //Console.WriteLine("blockIdx.x = " + gt.blockIdx.x.ToString());//  blockDim.x={1} threadIdx.x={2}", gt.blockDim.x, gt.threadIdx.x));
            float threshold = prices[absTx] + currTp;
            float openPrice = prices[absTx];
            float drawdown = 0;
            int step;
            for (step = absTx; step < N && prices[step] < threshold ; step++)
            {
                if (openPrice - prices[step] > drawdown)
                    drawdown = openPrice - prices[step];
            }
            if (step < N)
                durationArray[absTx] = step - absTx;
            else
                durationArray[absTx] = -1;
            drawdownArray[absTx] = drawdown;

        }

        internal static void MeasureSpeed(string name, int times, Action testAction, Func<double, string> addInfo = null)
        {
            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < times; i++)
                testAction();
            watch.Stop();
            double callsPerSecond = times * 1000.0 / (watch.ElapsedMilliseconds);
            string output = "\t" + name + " calls per second=> " + callsPerSecond + ";";
            if (addInfo != null)
                output += addInfo(callsPerSecond);
            Console.WriteLine(output);
        }
        internal static void MeasureTime(string name, int times, Action testAction, Func<double, string> addInfo = null)
        {
            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < times; i++)
                testAction();
            watch.Stop();
            string output = "\t" + name + " took=> " + watch.ElapsedMilliseconds + " millisencods;";
            Console.WriteLine(output);
        }
    }
}
