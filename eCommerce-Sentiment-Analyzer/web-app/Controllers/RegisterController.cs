    using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using _221229064_BitirmeProjesi.Entities;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using _221229064_BitirmeProjesi.Models;
using _221229064_BitirmeProjesi.Concrete;

namespace _221229064_BitirmeProjesi.Controllers
{
    public class RegisterController : Controller
    {
        private readonly SqlDbContext _dbContext;

        public RegisterController(SqlDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Index(MemberViewModel model)
        {
            if (ModelState.IsValid)
            {
                var newMember = new TblMembers
                {
                    Password = model.Password,
                    Email = model.Email,
                    FirstName = model.FirstName,
                    LastName = model.LastName,
                    RegistrationDate = DateTime.Now,
                    EmailCheck = 1,
                    IsActive = 1,
                    IsAdmin = 0
                };

                _dbContext.Members.Add(newMember);
                _dbContext.SaveChanges();

                return RedirectToAction("Index", "Home");
            }

            return View(model);
        }
    }
}

