using System.Diagnostics;
using System.IO;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class AutomaticLabelController : Controller
    {
        [Authorize(Roles = "Admin,User")]
        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [Authorize(Roles = "Admin,User")]
        [HttpPost]
        public IActionResult Index(IFormFile file, string modelType)
        {
            if (file == null || file.Length == 0)
            {
                TempData["ErrorMessage"] = "Lütfen bir dosya yükleyin.";
                return View();
            }

            string tempFilePath = Path.GetTempFileName(); // geçici dosya yolu
            string outputFileName = $"filled_csv_file_{modelType}.csv";
            string outputFilePath = Path.Combine(Path.GetTempPath(), outputFileName);
            string modelPath = $@"C:\Users\bilal\PycharmProjects\webScrapingProject\best_text_{modelType}_model_with_params.pth"; 

            try
            {
                string pythonScriptPath = $@"C:\Users\bilal\PycharmProjects\webScrapingProject\script_{modelType}.py";

                
                using (var stream = new FileStream(tempFilePath, FileMode.Create))
                {
                    file.CopyTo(stream);
                }

                var processStartInfo = new ProcessStartInfo
                {
                    FileName = @"C:\Users\bilal\PycharmProjects\webScrapingProject\.venv\Scripts\python.exe",
                    Arguments = $"\"{pythonScriptPath}\" \"{tempFilePath}\" \"{outputFilePath}\" \"{modelPath}\"",
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

                    
                    if (System.IO.File.Exists(outputFilePath))
                    {
                        byte[] fileBytes = System.IO.File.ReadAllBytes(outputFilePath);
                        return File(fileBytes, "application/octet-stream", outputFileName);
                    }
                    else
                    {
                        TempData["ErrorMessage"] = "Oluşan CSV dosyası bulunamadı.";
                        return View();
                    }
                }
            }
            catch (Exception ex)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu: " + ex.Message;
                return View();
            }
            finally
            {
               
                if (System.IO.File.Exists(tempFilePath))
                {
                    System.IO.File.Delete(tempFilePath);
                }
            }
        }
    }
}
