class ChangeColumnNameInAppointments < ActiveRecord::Migration[7.0]
  def change
    rename_column :appointments, :doctor_id, :user_id
  end
end
