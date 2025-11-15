using System.ComponentModel.DataAnnotations;

namespace _221229064_BitirmeProjesi.Entities
{
    public class TblMembers
    {
        [Key]
        public int MemberID { get; set; }
        public string ?Password { get; set; }
        public string ?Email { get; set; }
        public string ?FirstName { get; set; }
        public string ?LastName { get; set; }
        public DateTime RegistrationDate { get; set; }
        public int EmailCheck { get; set; }
        public int IsActive { get; set; }
        public int IsAdmin { get; set; }
    }
}
