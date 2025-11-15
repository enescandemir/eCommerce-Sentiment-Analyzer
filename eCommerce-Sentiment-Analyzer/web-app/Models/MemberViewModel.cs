using System.ComponentModel.DataAnnotations;

namespace _221229064_BitirmeProjesi.Models
{
    public class MemberViewModel
    {
        [Required]
        public string Password { get; set; }

        [Required]
        [EmailAddress]
        public string Email { get; set; }

        [Required]
        public string FirstName { get; set; }

        [Required]
        public string LastName { get; set; }
    }
}
