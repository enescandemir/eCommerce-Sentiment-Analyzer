using System.ComponentModel.DataAnnotations;

namespace _221229064_BitirmeProjesi.Entities
{
    public class TblProcessedComments
    {
        [Key]
        public int Comment_ID { get; set; }
        public int Product_ID { get; set; }
        public string? Comment_Context { get; set; }
        public int? Comment_Context_Ticket { get; set; }
        public int? LabeledBy { get; set; }
    }
}
