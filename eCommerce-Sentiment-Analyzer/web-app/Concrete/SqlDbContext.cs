using System;
using Microsoft.EntityFrameworkCore;
using _221229064_BitirmeProjesi.Entities;

namespace _221229064_BitirmeProjesi.Concrete
{
    public class SqlDbContext : DbContext
    {

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!optionsBuilder.IsConfigured)
            {
                var connectionString = Environment.GetEnvironmentVariable("DB_CONNECTION_STRING");
                optionsBuilder.UseSqlServer(connectionString);
            }
        }

        public DbSet<TblMembers> Members { get; set; }
        public DbSet<TblProducts> Products { get; set; }
        public DbSet<TblComments> Comments { get; set; }
        public DbSet<TblProcessedComments> ProcessedComments { get; set; }

    }
}
