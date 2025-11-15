using System.Diagnostics;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class DataExtractionController : Controller
    {
        [Authorize(Roles = "Admin,User")]
        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [Authorize(Roles = "Admin,User")]
        [HttpPost]
        public IActionResult Index(string site, string link)
        {
            if (string.IsNullOrEmpty(link))
            {
                TempData["ErrorMessage"] = "Lütfen bir link giriniz.";
                return View();
            }

            string pythonScriptPath;

            if (site == "Trendyol")
            {
                pythonScriptPath = @"C:\Users\bilal\PycharmProjects\webScrapingProject\webscraping.py";
            }
            else if (site == "Hepsiburada")
            {
                pythonScriptPath = @"C:\Users\bilal\PycharmProjects\webScrapingProject\webscrapinghb.py";
            }
            else
            {
                TempData["ErrorMessage"] = "Geçersiz site seçimi.";
                return View();
            }

            var processStartInfo = new ProcessStartInfo
            {
                FileName = @"C:\Users\bilal\PycharmProjects\webScrapingProject\.venv\Scripts\python.exe", 
                Arguments = $"\"{pythonScriptPath}\" \"{link}\"",
                RedirectStandardOutput = true, // ciktilari almak icin
                RedirectStandardError = true, // hatalari almak icin
                UseShellExecute = false, // shell devre disi
                CreateNoWindow = true // yeni pencere acmamak icin
            };

            try
            {
                using (var process = Process.Start(processStartInfo))
                {
                    // ciktiyi okumak icin
                    string result = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();

                    // process bitmesini beklemek icin
                    process.WaitForExit();

                    if (!string.IsNullOrEmpty(error))
                    {
                        TempData["ErrorMessage"] = "Python scripti çalıştırılırken bir hata oluştu: " + error;
                    }
                    else
                    {
                        TempData["SuccessMessage"] = "Python scripti başarıyla çalıştırıldı. Çıktı: " + result;
                    }
                }
            }
            catch (Exception ex)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu: " + ex.Message;
            }

            return View();
        }

        [Authorize(Roles = "Admin,User")]
        [HttpPost]
        public IActionResult DownloadCsv()
        {
            string pythonScriptPath = @"C:\Users\bilal\PycharmProjects\webScrapingProject\csv_export.py";

            var processStartInfo = new ProcessStartInfo
            {
                FileName = @"C:\Users\bilal\PycharmProjects\webScrapingProject\.venv\Scripts\python.exe", 
                Arguments = $"\"{pythonScriptPath}\"", 
                RedirectStandardOutput = true, 
                RedirectStandardError = true, 
                UseShellExecute = false, 
                CreateNoWindow = true 
            };

            try
            {
                using (var process = Process.Start(processStartInfo))
                {
                    string result = process.StandardOutput.ReadToEnd(); // pythondan gelen csv verisi
                    string error = process.StandardError.ReadToEnd();

                    process.WaitForExit();

                    if (!string.IsNullOrEmpty(error))
                    {
                        TempData["ErrorMessage"] = "CSV dosyası oluşturulurken bir hata oluştu: " + error;
                        return RedirectToAction("Index");
                    }
                    else
                    {
                        byte[] fileBytes = System.Text.Encoding.UTF8.GetBytes(result);  // csv verisini dosya olarak indirilmek üzere dondurme islemi
                        string fileName = "comments.csv";

                        return File(fileBytes, "text/csv", fileName);
                    }
                }
            }
            catch (Exception ex)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu: " + ex.Message;
                return RedirectToAction("Index");
            }
        }

    }
}
