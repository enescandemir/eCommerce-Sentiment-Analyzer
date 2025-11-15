using System.ComponentModel.DataAnnotations;

namespace _221229064_BitirmeProjesi.Entities
{
    public class TblProducts
    {
        [Key]
        public int Product_ID { get; set; }
        public string ?Product_Name { get; set; }
        public string ?Product_Brand { get; set; }
        public string ?Product_Link { get; set; }
        public string ?Product_Price { get; set; }
        public string ?Product_Star_Rating { get; set; }
        public string ?Product_Image_Url { get; set; }
    }
}
