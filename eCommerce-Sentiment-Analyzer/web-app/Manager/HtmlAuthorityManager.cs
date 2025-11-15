using System;
using System.Security.Claims;
using Microsoft.AspNetCore.Mvc;
using _221229064_BitirmeProjesi.Enums;

namespace _221229064_BitirmeProjesi.Manager
{
	public class HtmlAuthorityManager
	{
        private readonly IHttpContextAccessor _httpContextAccessor;

        public HtmlAuthorityManager(IHttpContextAccessor httpContextAccessor)
        {
            _httpContextAccessor = httpContextAccessor;
        }


        public string GetDisplayVariable(RoleEnum role)
        {
            ClaimsPrincipal? currentUser = _httpContextAccessor.HttpContext?.User;
            string? claimRole = currentUser?.FindFirst(ClaimTypes.Role)?.Value;
            RoleEnum roleEnum;

            if (currentUser == null || claimRole == null)
                roleEnum = RoleEnum.Null;
            else
                roleEnum = GetRoleEnum(claimRole);

            if (roleEnum == role)
                return "block";
            else
                return "none";
        }


        public string GetDisplayVariable(RoleEnum roleEnum1, RoleEnum roleEnum2)
        {
            ClaimsPrincipal? currentUser = _httpContextAccessor.HttpContext?.User;
            string? claimRole = currentUser?.FindFirst(ClaimTypes.Role)?.Value;
            RoleEnum roleEnum;

            if (currentUser == null || claimRole == null)
                roleEnum = RoleEnum.Null;
            else
                roleEnum = GetRoleEnum(claimRole);


            if (roleEnum == roleEnum1 || roleEnum == roleEnum2)
                return "block";
            else
                return "none";
        }


        public string onAuthenticate()
        {
            ClaimsPrincipal? currentUser = _httpContextAccessor.HttpContext?.User;
            string? claimRole = currentUser?.FindFirst(ClaimTypes.Role)?.Value;
            RoleEnum roleEnum;

            if (currentUser == null || claimRole == null)
                roleEnum = RoleEnum.Null;
            else
                roleEnum = GetRoleEnum(claimRole);

            if (roleEnum == RoleEnum.Null)
                return "block";
            else
                return "none";
        }





        public RoleEnum GetRoleEnum(string claimRole)
        {
            if (claimRole == RoleEnum.Admin.ToString()) return RoleEnum.Admin;
            else if (claimRole == RoleEnum.User.ToString()) return RoleEnum.User;
            else return RoleEnum.Null;
        }

    }
}

