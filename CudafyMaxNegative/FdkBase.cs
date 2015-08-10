using SoftFX.Extended;
using SoftFX.Extended.Events;
using SoftFX.Extended.Storage;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace CudafyMaxNegative
{
    public class FdkBase : IDisposable
    {
        public FdkBase()
        {
        }
        public void Login(string address, string username, string password)
        {
               
            #region EnsureDirectoryCreating
            string CommonPath = Assembly.GetEntryAssembly() != null ? Path.GetDirectoryName(Assembly.GetEntryAssembly().Location) : string.Empty;
            string logPath = Path.Combine(CommonPath, "Logs");
            string storagePath = Path.Combine(CommonPath, "Storage");
            if (!Directory.Exists(logPath))
                Directory.CreateDirectory(logPath);

            if (!Directory.Exists(storagePath))
                Directory.CreateDirectory(storagePath);
            #endregion

            FixConnectionStringBuilder builder = new FixConnectionStringBuilder
            {
                SecureConnection = false,
                Port = 5001,
                //ExcludeMessagesFromLogs = "W"
                Address = address,
                FixLogDirectory = logPath,
                FixEventsFileName = string.Format("FIX_{0}.feed.events.log", username),
                FixMessagesFileName = string.Format("FIX_{0}.feed.messages.log", username),

                Username = username,
                Password = password
            };

            this.Feed = new DataFeed
            {
                SynchOperationTimeout = 60000
            };

            this.Feed.Initialize(builder.ToString());
            this.Feed.Logon += this.OnLogon;
            this.Feed.Logout += this.OnLogout;
            this.Feed.Notify += this.OnNotify;

            this.Feed.SessionInfo += this.OnSessionInfo;
            this.Feed.SymbolInfo += this.OnSymbolInfo;
            this.Feed.CurrencyInfo += this.OnCurrencyInfo;

            this.Storage = new DataFeedStorage(storagePath, StorageProvider.Ntfs, this.Feed, true);
            
            if (!this.Feed.Start(this.Feed.SynchOperationTimeout))
            {
                Console.ReadKey();
                throw new TimeoutException("Timeout of logon waiting has been reached");
            }
         }
        #region Event Handlers

        void OnLogon(object sender, LogonEventArgs e)
        {
            Console.WriteLine("OnLogon(): {0}", e);
        }

        void OnLogout(object sender, LogoutEventArgs e)
        {
            Console.WriteLine("OnLogout(): {0}", e);
        }

        void OnSymbolInfo(object sender, SymbolInfoEventArgs e)
        {
        }

        void OnCurrencyInfo(object sender, CurrencyInfoEventArgs e)
        {
        }

        void OnSessionInfo(object sender, SessionInfoEventArgs e)
        {
        }

        void OnNotify(object sender, NotificationEventArgs e)
        {
            Console.WriteLine("OnNotify(): {0}", e);
        }

        #endregion

        protected DataFeed Feed { get; private set; }
        public DataFeedStorage Storage { get; private set; }


        public void Dispose()
        {
            if (this.Feed != null)
            {
                this.Feed.Stop();
                this.Feed.Dispose();
                this.Feed = null;
            }

            if (this.Storage != null)
            {
                this.Storage.Dispose();
                this.Storage = null;
            }
        }
    }
}
