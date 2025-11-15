using Microsoft.AspNetCore.Mvc;
using _221229064_BitirmeProjesi.Entities;
using _221229064_BitirmeProjesi.Models;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using _221229064_BitirmeProjesi.Concrete;
using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using System.Diagnostics;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class ManualLabelController : Controller
    {
        private readonly SqlDbContext _dbContext;

        public ManualLabelController(SqlDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [HttpGet]
        public async Task<IActionResult> Index()
        {
            var comments = _dbContext.ProcessedComments.ToList(); // tum yorumlari getirme

            // Kullanıcı yetki kontrolü:
            string? email = User.FindFirst(ClaimTypes.Email)?.Value;
            var user = await _dbContext.Members.FirstOrDefaultAsync(x => x.Email == email);

            if (user == null || user.IsAdmin != 1) {
                foreach (var comment in comments)
                {
                    comment.LabeledBy = null;
                }
            }

            return View(comments);
        }

        // ajax etiketleme islemi
        [HttpPost]
        public async Task<IActionResult> LabelComment(int commentId, int label)
        {
            var comment = await _dbContext.ProcessedComments.FirstOrDefaultAsync(c => c.Comment_ID == commentId);

            if (comment != null)
            {
                comment.Comment_Context_Ticket = label;

                // kullanici idsini int olarak cekip labeledby atama
                var userIdClaim = User.FindFirst(ClaimTypes.Email)?.Value;
                if (userIdClaim != null)
                {
                    var user = await _dbContext.Members.FirstOrDefaultAsync(m => m.Email == userIdClaim);
                    comment.LabeledBy = user?.MemberID ?? 0;
                }
                else
                {
                    comment.LabeledBy = 0;
                }

                _dbContext.Update(comment);
                await _dbContext.SaveChangesAsync();
            }

            return Json(new { success = true });
        }

        [Authorize(Roles = "Admin,User")]
        [HttpPost]
        public IActionResult DownloadProcessedCsv()
        {
            string pythonScriptPath = @"C:\Users\bilal\PycharmProjects\webScrapingProject\processed_csv_export.py";

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
                        string fileName = "processed_comments.csv";

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
