using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Diagnostics;
using SoftFX.Extended;
using System.Reflection;
using System.Globalization;

namespace CudafyMaxNegative
{
    class Program
    {
        static void Main(string[] args)
        {
            Library.Initialize();
            InitGPU();

            List<float> pricesList = new List<float>();
            List<string> datesList = new List<string>();
            
            Bar[] bars = Quotes.Get("EURUSD", 
                DateTime.Parse("07/01/2015", CultureInfo.InvariantCulture),
                DateTime.Parse("07/30/2015", CultureInfo.InvariantCulture)
            );

            pricesList.AddRange(bars.Select(p => (float)p.Open));
            datesList.AddRange(bars.Select(p => p.From.ToString()));

            Console.WriteLine("Starting analyses of {0} items.", pricesList.Count);

            using( StreamWriter sw = new StreamWriter("_Cudafy.csv"))
            {
                sw.WriteLine("Date, Tp, Drawdown,BarDuration");
                MeasureTime("GPGPU seq", 1, ()=>
                    {
                        CalculateAll(sw, pricesList.ToArray(), datesList.ToArray(), 0.003f, 0.005f, 0.001f);
                    });
            }

        }



        static GPGPU InitGPU()
        {
            try
            {
                CudafyModes.Target = eGPUType.Cuda; // To use OpenCL, change this enum
                CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;

            }
            catch (Exception ex)
            {
                var ex2 = ex.InnerException.InnerException;
                if (ex2 is System.Reflection.ReflectionTypeLoadException)
                {
                    var typeLoadException = ex2 as ReflectionTypeLoadException;
                    var loaderExceptions = typeLoadException.LoaderExceptions;
                    int k = 9;
                }
            }
            CudafyModes.Target = eGPUType.Cuda; // To use OpenCL, change this enum
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
                int BLOCK_SIZE = 64;
                gpu.LaunchAsync((N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 1, "calculate", dev_prices, currTp, N, dev_drawdown, dev_duration);

                gpu.Synchronize();

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

            float threshold = prices[absTx] + currTp;
            float openPrice = prices[absTx];
            float drawdown = 0;
            int step;
            for (step = absTx; step < N-1  ; step++)
            {
                float currPrice = prices[step];
                if (currPrice >= threshold)
                    break;
                if (openPrice - currPrice > drawdown)
                    drawdown = openPrice - currPrice;
            }
            //Console.WriteLine("absTx = %d step=%d drawdown=%f", absTx, step, drawdown);//  blockDim.x={1} threadIdx.x={2}", gt.blockDim.x, gt.threadIdx.x));

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
