using _221229064_BitirmeProjesi.Models;
using Microsoft.AspNetCore.Mvc;
using _221229064_BitirmeProjesi.Concrete;
using _221229064_BitirmeProjesi.Entities;
using Microsoft.AspNetCore.Authentication;
using System.Security.Claims;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authorization;
using _221229064_BitirmeProjesi.Enums;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class LoginController : Controller
    {
        private readonly SqlDbContext _dbContext;

        public LoginController(SqlDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Index(TblMembers member)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    TempData["ErrorMessage"] = "Veriler doğru iletilmedi!";
                    return RedirectToAction("Index", "Login");
                }
                if (member == null)
                {
                    TempData["ErrorMessage"] = "Herhangi bir veri girişi yapılmadı!";
                    return RedirectToAction("Index", "Login");
                }

                TblMembers? memberData = _dbContext.Members.FirstOrDefault(m => m.Email == member.Email);
                if (memberData != null)
                {
                    if (memberData.Password == member.Password)
                    {
                        var claims = new List<Claim>
                        {
                            new Claim(ClaimTypes.Email, member.Email),
                            new Claim(ClaimTypes.Role, (memberData.IsAdmin == 1) ? RoleEnum.Admin.ToString() : RoleEnum.User.ToString()),
                            new Claim(ClaimTypes.Name, memberData.FirstName + " " + memberData.LastName)
                        };
                        var userIdentity = new ClaimsIdentity(claims, " ");
                        var authProperties = new AuthenticationProperties
                        {
                            IsPersistent = true, // Oturumun kalıcı olmasını sağlar (true)
                            ExpiresUtc = DateTimeOffset.UtcNow.AddDays(7) // Opsiyonel: Çerezin ne kadar süreyle geçerli olacağını belirler
                        };
                        ClaimsPrincipal principal = new ClaimsPrincipal(userIdentity);
                        await HttpContext.SignInAsync(principal, authProperties);
                        return RedirectToAction("Index", "Home");
                    }
                    else
                    {
                        TempData["ErrorMessage"] = "Şifre Hatalı!";
                        return RedirectToAction("Index", "Login");
                    }
                }
                else
                {
                    TempData["ErrorMessage"] = "Kullanıcı bulunamadı!";
                    return RedirectToAction("Index", "Login");
                }

            }
            catch (Exception ex)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu!";
                Console.WriteLine("Bir hata oluştu: " + ex);
                return RedirectToAction("Index", "Login");
            }
        }


        [Authorize]
        [HttpPost]
        public async Task<IActionResult> Logout()
        {
            await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            return RedirectToAction("Index", "Login");
        }
    }
}
