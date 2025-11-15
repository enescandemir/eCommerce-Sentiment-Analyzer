using System.Diagnostics;
using System.IO;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class DataPreProcessController : Controller
    {
        [Authorize(Roles = "Admin,User")]
        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [Authorize(Roles = "Admin,User")]
        [HttpPost]
        public IActionResult Index(IFormFile file, bool Lowercase = false, bool RemoveNumbers = false, bool RemovePunctuation = false, bool RemoveEmojis = false, bool CharacterRepetitionReduction = false, bool StopwordRemoval = false)
        {
            if (file == null || file.Length == 0)
            {
                TempData["ErrorMessage"] = "Lütfen bir dosya yükleyin.";
                return View();
            }

            string tempFilePath = Path.GetTempFileName(); // gecici dosya yolu

            try
            {
                string pythonScriptPath = @"C:\Users\bilal\PycharmProjects\webScrapingProject\dataprocess.py";

                // yuklenen dosyayi gecici bir yola kaydetme
                using (var stream = new FileStream(tempFilePath, FileMode.Create))
                {
                    file.CopyTo(stream);
                }

                var processStartInfo = new ProcessStartInfo
                {
                    FileName = @"C:\Users\bilal\PycharmProjects\webScrapingProject\.venv\Scripts\python.exe",
                    Arguments = $"\"{pythonScriptPath}\" \"{tempFilePath}\" \"{Lowercase.ToString().ToLower()}\" \"{RemoveNumbers.ToString().ToLower()}\" \"{RemovePunctuation.ToString().ToLower()}\" \"{RemoveEmojis.ToString().ToLower()}\" \"{CharacterRepetitionReduction.ToString().ToLower()}\" \"{StopwordRemoval.ToString().ToLower()}\"",

                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (var process = Process.Start(processStartInfo))
                {
                    string result = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();

                    process.WaitForExit();

                    if (!string.IsNullOrEmpty(error))
                    {
                        TempData["ErrorMessage"] = "Python scripti çalıştırılırken bir hata oluştu: " + error;
                        return View();
                    }

                    TempData["SuccessMessage"] = "Veriler başarıyla işlendi ve veritabanına kaydedildi.";
                    return View();
                }
            }
            catch (Exception ex)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu: " + ex.Message;
                return View();
            }
            finally
            {
                // gecici dosyayi sil
                if (System.IO.File.Exists(tempFilePath))
                {
                    System.IO.File.Delete(tempFilePath);
                }
            }
        }
    }
}
