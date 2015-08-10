using SoftFX.Extended;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudafyMaxNegative
{
    public static class Quotes
    {
        public static Bar[] Get(string symbol, DateTime from, DateTime to)
        {
            using (FdkBase fdk = new FdkBase())
            {

                fdk.Login("ttlive.fxopen.com", "800042-readonly", "wcE8dCdxaSFR");
                return fdk.Storage.Online.GetBars(symbol, PriceType.Ask, BarPeriod.S1, from, to);
            }
        }
    }
}
