using Microsoft.AspNetCore.Mvc;
using _221229064_BitirmeProjesi.Models;
using _221229064_BitirmeProjesi.Concrete;
using _221229064_BitirmeProjesi.Entities;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;
using Microsoft.EntityFrameworkCore;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class PanelController : Controller
    {
        private readonly SqlDbContext _dbContext;

        public PanelController(SqlDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [Authorize(Roles = "Admin")]
        [HttpGet]
        public IActionResult Index()
        {
            try
            {
                List<TblMembers>? members = _dbContext.Members.ToList();

                if (members != null && members.Count > 0)
                {
                    return View(members);
                }
                else
                {
                    return RedirectToAction("Index", "Home");
                }
            }
            catch (Exception)
            {
                return RedirectToAction("Index", "Home");
            }
        }

        [Authorize(Roles = "Admin")]
        [HttpPost]
        public async Task<IActionResult> DeleteAccount(int MemberID)
        {
            try
            {
                string? email = User.FindFirst(ClaimTypes.Email)?.Value;

                if (email != null && email.Length > 0)
                {
                    TblMembers? member = await _dbContext.Members.FirstOrDefaultAsync(user => user.MemberID == MemberID);

                    if (member == null)
                    {
                        TempData["ErrorMessage"] = "Bir hata oluştu!";
                        return RedirectToAction("Index", "Panel");
                    }

                    if (member.Email == email)
                    {
                        TempData["ErrorMessage"] = "Kendinizi Silemezsiniz!";
                        return RedirectToAction("Index", "Panel");
                    }

                    _dbContext.Members.Remove(member);
                    await _dbContext.SaveChangesAsync();

                    return RedirectToAction("Index", "Panel");
                }
                else
                {
                    TempData["ErrorMessage"] = "Bir hata oluştu!";
                    return RedirectToAction("Index", "Panel");
                }
            }
            catch (Exception)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu!";
                return RedirectToAction("Index", "Panel");
            }
        }


        [Authorize(Roles = "Admin")]
        [HttpPost]
        public async Task<IActionResult> MakeAdmin(int MemberID)
        {
            try
            {
                string? email = User.FindFirst(ClaimTypes.Email)?.Value;

                if (email != null && email.Length > 0)
                {
                    TblMembers? member = await _dbContext.Members.FirstOrDefaultAsync(user => user.MemberID == MemberID);

                    if (member == null)
                    {
                        TempData["ErrorMessage"] = "Bir hata oluştu!";
                        return RedirectToAction("Index", "Panel");
                    }

                    if (member.Email == email)
                    {
                        TempData["ErrorMessage"] = "Kendinizi Admin yapamazsınız!";
                        return RedirectToAction("Index", "Panel");
                    }

                    member.IsAdmin = 1;
                    await _dbContext.SaveChangesAsync();

                    return RedirectToAction("Index", "Panel");
                }
                else
                {
                    TempData["ErrorMessage"] = "Bir hata oluştu!";
                    return RedirectToAction("Index", "Panel");
                }
            }
            catch (Exception)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu!";
                return RedirectToAction("Index", "Panel");
            }
        }


        [Authorize(Roles = "Admin")]
        [HttpPost]
        public async Task<IActionResult> RemoveAdmin(int MemberID)
        {
            try
            {
                string? email = User.FindFirst(ClaimTypes.Email)?.Value;

                if (email != null && email.Length > 0)
                {
                    TblMembers? member = await _dbContext.Members.FirstOrDefaultAsync(user => user.MemberID == MemberID);

                    if (member == null)
                    {
                        TempData["ErrorMessage"] = "Bir hata oluştu!";
                        return RedirectToAction("Index", "Panel");
                    }

                    if (member.Email == email)
                    {
                        TempData["ErrorMessage"] = "Kendi Admin yetkinizi kaldıramazsınız!";
                        return RedirectToAction("Index", "Panel");
                    }

                    member.IsAdmin = 0;
                    await _dbContext.SaveChangesAsync();

                    return RedirectToAction("Index", "Panel");
                }
                else
                {
                    TempData["ErrorMessage"] = "Bir hata oluştu!";
                    return RedirectToAction("Index", "Panel");
                }
            }
            catch (Exception)
            {
                TempData["ErrorMessage"] = "Bir hata oluştu!";
                return RedirectToAction("Index", "Panel");
            }
        }
    }
}
