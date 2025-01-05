class CreatePatients < ActiveRecord::Migration[7.0]
  def change
    create_table :patients do |t|
      t.text :first_name
      t.string :last_name
      t.date :dob
      t.text :gender
      t.text :address
      t.string :phone
      t.string :email

      t.timestamps
    end
  end
end
